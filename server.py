"""同声传译演示 - 全本地版（Apple Silicon / MLX）。

浏览器只负责收音、切句和播放，其余都在本机跑，不连任何云服务：
- 说话过程中：流式识别（sherpa-onnx zipformer，CPU）出实时字幕、判断中英文，并用 Qwen 出"草稿翻译"
- 一句说完：Whisper（GPU）精确识别 → Qwen（GPU）正式翻译，替换草稿；正式任务一来，草稿立刻让出 GPU
- 朗读：Kokoro（CPU）
运行:  .venv312/bin/python server.py   然后用 Chrome 打开 http://localhost:8765
只看界面（不加载模型）:  MOCK=1 .venv312/bin/python server.py
"""
import io
import json
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

warnings.filterwarnings("ignore")
# 默认离线：模型都已下载到本机。不设的话，每次加载都会先联网检查更新；
# 网络连不上 Hugging Face 时会一直等到超时，启动要 4–5 分钟（实测 280 秒）。setup.sh 下载时会临时关掉它。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PORT = int(os.environ.get("PORT", "8765"))
MOCK = os.environ.get("MOCK") == "1"
_HERE = Path(__file__).parent
# 识别模型默认用本地 4 位量化版（setup.sh 生成）：实测准确率、速度和原版一样，GPU 内存从 2.1GB 降到 1.1GB
_Q4 = _HERE / "models" / "whisper-large-v3-turbo-q4"
ASR_MODEL = os.environ.get("ASR_MODEL", str(_Q4) if _Q4.exists() else "mlx-community/whisper-large-v3-turbo")
MT_MODEL = os.environ.get("MT_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
MT_ADAPTER = os.environ.get("MT_ADAPTER", "")
HERE = Path(__file__).parent
TTS_MODEL = HERE / "models" / "kokoro-v1.0.onnx"
TTS_VOICES = HERE / "models" / "voices-v1.0.bin"
DEFAULT_VOICE = os.environ.get("VOICE", "af_heart")

# GPU 上同一时间只跑一个模型，避免 MLX 并发出问题
gpu = threading.Lock()
# 朗读跑在 CPU 上（onnxruntime），和 GPU 上的识别/翻译互不阻塞。
# 固定用同一个线程：onnxruntime 在每个新线程上第一次运行要多花 ~2 秒
tts_worker = ThreadPoolExecutor(max_workers=1)
kokoro = None
# 流式识别同理固定一个线程
STREAM_ASR_DIR = HERE / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
stream_worker = ThreadPoolExecutor(max_workers=1)
recognizer = None
streams = {}  # 每句话一个流：sid -> [stream, 最后访问时间]
streams_lock = threading.Lock()

# 正式任务（一句话说完后的识别/翻译）在排队时，草稿翻译要立刻让出 GPU
finals_waiting = 0
finals_lock = threading.Lock()


class FinalTask:
    """正式任务：先登记"有正式任务在等"，再去抢 GPU。"""

    def __enter__(self):
        global finals_waiting
        with finals_lock:
            finals_waiting += 1
        gpu.acquire()
        with finals_lock:
            finals_waiting -= 1

    def __exit__(self, *exc):
        free_gpu_cache()
        gpu.release()

# --- 正式翻译的规则：针对会议里实测出错最多的口语现象 ---
SYSTEM = """You are a live interpreter on a call between a Chinese speaker and an English speaker. Translate each utterance into natural, spoken language.
Rules:
1. Keep who is speaking: 我 = I, 你 = you. Never change the subject of the sentence.
2. Terms of address are not the subject. 老婆, 老公, 宝贝, 亲爱的, 妈, 兄弟 or a name at the start of a sentence (often after 喂) is the speaker calling the listener: 老婆你听得到吗 -> Honey, can you hear me?
3. Spoken Chinese repeats words and uses fillers (喂喂喂, 那个, 就是, 然后, 那个什么, 嗯, 啊). Say repeated calls once, turn fillers into light English fillers or drop them, and keep only the final meaning of self-corrections.
4. Translate meaning and tone, not word for word. Give slang and idioms an English equivalent.
5. The text comes from speech recognition and may contain homophone mistakes. Infer the intended words.
6. Always translate into the other language, even single words and names of languages. Never copy the input back, and never leave untranslated words in the output.
7. Output only the translation of the last message, with no notes."""
# 方向指令写死"只输出某种语言"：以前英译中有时把英文原样吐回来，中译英里夹着没译的中文词
DIRECTIONS = {
    "en2zh": "把下面的英文翻译成简体中文，只输出中文：",
    "zh2en": "Translate the following Chinese into English. Output English only:",
}
# 示例：都是会议里真实出过错的说法
FINAL_SHOTS = [
    ("zh2en", "喂喂喂老婆你听得到我声音吗", "Hello? Honey, can you hear me?"),
    ("zh2en", "老婆厉害吧，我现在搞定了", "Pretty cool, right, honey? I've got it working now."),
    ("zh2en", "还有就是那个什么，商业化很麻烦。", "And, um, what's it called... commercializing it is a real hassle."),
    ("zh2en", "当然可以哦，谁有流量谁说了算啊。", "Of course! Whoever has the traffic calls the shots."),
    ("zh2en", "宝贝宝贝宝贝，过来一下", "Babe, come here for a sec."),
    ("en2zh", "Babe, can you hear me?", "宝贝，你听得到我吗？"),
    ("en2zh", "That's awesome, but it sounds too robotic.", "太厉害了，不过听起来太像机器了。"),
    ("en2zh", "Spanish.", "西班牙语。"),
    ("en2zh", "I'm so dead.", "我完了。"),
]

# --- 热词表（hotwords.txt，所有场景通用）+ 当前场景（scenes/<名字>.txt）：改完保存即生效 ---
HOTWORDS_FILE = HERE / "hotwords.txt"
SCENES_DIR = HERE / "scenes"
current_scene = os.environ.get("SCENE", "")
_hot = {"key": None, "words": [], "fixes": [], "glossary": [], "styles": []}


def list_scenes():
    return sorted(p.stem for p in SCENES_DIR.glob("*.txt")) if SCENES_DIR.is_dir() else []


def _mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def hotwords():
    """读热词表 + 当前场景：优先认出的词、自动改正、固定译法、翻译语气。文件或场景变了才重新读。"""
    scene_file = SCENES_DIR / f"{current_scene}.txt" if current_scene else None
    key = (_mtime(HOTWORDS_FILE), current_scene, scene_file and _mtime(scene_file))
    if key != _hot["key"]:
        words, fixes, glossary, styles = [], [], [], []
        lines = []
        for f in (HOTWORDS_FILE, scene_file):
            if f and f.exists():
                lines += f.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@语气") or line.lower().startswith("@style"):
                styles.append(re.split(r"[:：]", line, 1)[-1].strip())
            elif "=>" in line:
                a, b = (x.strip() for x in line.split("=>", 1))
                if a and b:
                    fixes.append((a, b))
            elif "=" in line:
                a, b = (x.strip() for x in line.split("=", 1))
                if a and b:
                    glossary.append((a, b))
            else:
                words.append(line)
        _hot.update(key=key, words=words, fixes=fixes, glossary=glossary, styles=styles)
        caches.pop("final", None)  # 固定译法/语气变了，翻译规则的缓存要重建
        log(f"热词表已加载（场景：{current_scene or '通用'}）：{len(words)} 个词，{len(fixes)} 条改正，"
            f"{len(glossary)} 条固定译法，{len(styles)} 条语气")
    return _hot


def apply_fixes(text):
    for a, b in hotwords()["fixes"]:
        text = text.replace(a, b)
    return text


def system_prompt():
    h = hotwords()
    p = SYSTEM
    if h["styles"]:
        p += "\nTone for this conversation: " + " ".join(h["styles"])
    if h["glossary"]:
        p += "\nAlways use these fixed translations:\n" + "\n".join(f"- {a} = {b}" for a, b in h["glossary"])
    return p


def asr_prompt(lang):
    """Whisper 的提示：基础提示 + 热词，帮它认出专有名词。"""
    words = hotwords()["words"]
    if lang == "zh":
        return ASR_PROMPT["zh"] + (f"其中提到：{'、'.join(words)}。" if words else "")
    en = [w for w in words if re.fullmatch(r"[A-Za-z0-9 .+#-]+", w)]
    return ("Glossary: " + ", ".join(en) + ".") if en else None
# 给 Whisper 的提示：中文用一句普通口语，让它倾向输出简体和标点。
# 以前写的是"使用简体中文……AI、Python、React……"，安静片段上 Whisper 会把提示词原样当成识别结果吐出来。
# 英文不需要提示。
ASR_PROMPT = {"zh": "以下是普通话的对话记录。", "en": None}  # 选一句没人会在对话里说的话，泄漏了也好认
# Whisper 在噪音上常见的幻觉输出
HALLUCINATIONS = re.compile(
    r"^(thank you\.?|thanks for watching!?|you\.?|bye\.?|字幕.*|请不吝点赞.*|谢谢观看.*|\.+)$", re.I
)
HALLUCINATION_TAIL = re.compile(
    r"(请不吝点赞.*|字幕由.*|字幕志愿者.*|谢谢观看.*|明镜.{0,6}(支持|订阅).*|欢迎订阅.*|点赞.{0,4}订阅.*|thanks for watching.*)$", re.I
)
# 这两种文字不该出现：俄文、韩文、日文假名、阿拉伯文、天城文、泰文——出现就是乱码
FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ가-힯぀-ヿ؀-ۿऀ-ॿ฀-๿]")
SILERO_VAD = HERE / "models" / "silero_vad.onnx"
vad = None
MIN_SPEECH_SEC = 0.3

np = mlx_whisper = model = tokenizer = stream_generate = None


def load_models():
    global np, mlx_whisper, model, tokenizer, stream_generate
    import numpy as np  # noqa: F811
    import mlx_whisper  # noqa: F811
    from mlx_lm import load, stream_generate  # noqa: F811

    import mlx.core as mx

    globals().update(np=np, mlx_whisper=mlx_whisper, stream_generate=stream_generate)
    # MLX 默认会把用过的显存留作缓存不还，跑一会儿就涨到 8GB+，16GB 的 Mac 会被挤进交换区、一句话识别要 6–10 秒
    mx.set_cache_limit(int(os.environ.get("MLX_CACHE_MB", "512")) * 1024 * 1024)
    t = time.time()
    # MT_ADAPTER：用纠错数据训练出的 LoRA（corrections.py export 之后用 mlx_lm lora 训练得到的目录）
    model, tokenizer = load(MT_MODEL, adapter_path=MT_ADAPTER or None)
    build_caches()
    warmup()
    global kokoro
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(str(TTS_MODEL), str(TTS_VOICES))
    synthesize("Ready.", DEFAULT_VOICE)  # 预热
    global recognizer
    import sherpa_onnx

    d = STREAM_ASR_DIR
    recognizer = stream_worker.submit(
        sherpa_onnx.OnlineRecognizer.from_transducer,
        tokens=str(d / "tokens.txt"),
        encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(d / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(d / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    ).result()
    stream_feed("warm", np.zeros(16000, dtype=np.float32).tobytes())  # 预热
    stream_close("warm")
    global vad
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(SILERO_VAD)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.min_speech_duration = 0.15
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    vad = stream_worker.submit(sherpa_onnx.VoiceActivityDetector, cfg, buffer_size_in_seconds=60).result()
    speech_seconds(np.zeros(16000, dtype=np.float32).tobytes())  # 预热
    print(f"模型已加载（{time.time() - t:.1f}s）", flush=True)


def detect_lang(text):
    """看实时字幕里汉字多还是英文单词多。"""
    cjk = len(re.findall(r"[一-鿿]", text))
    words = len(re.findall(r"[A-Za-z]+", text))
    if cjk == 0 and words == 0:
        return None
    return "zh" if cjk >= words else "en"


def tidy_stream_text(text):
    """zipformer 英文输出全是大写、没标点，转成正常大小写方便看。"""
    text = text.strip()
    if re.search(r"[A-Z]{2}", text):
        text = text.lower()
        text = re.sub(r"\bi\b", "I", text)
        text = text[:1].upper() + text[1:]
    return apply_fixes(text)


def stream_feed(sid, pcm_bytes):
    """把一小段音频喂给这句话的流式识别，返回目前的识别结果。"""
    if MOCK:
        return "(mock live caption)"

    def work():
        with streams_lock:
            entry = streams.get(sid)
            if entry is None:
                entry = streams[sid] = [recognizer.create_stream(), 0]
            entry[1] = time.time()
            # 顺手清理一分钟没动静的流
            for k in [k for k, v in streams.items() if time.time() - v[1] > 60]:
                del streams[k]
        s = entry[0]
        s.accept_waveform(16000, np.frombuffer(pcm_bytes, dtype=np.float32))
        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
        return tidy_stream_text(recognizer.get_result(s))

    return stream_worker.submit(work).result()


def stream_text(sid):
    if MOCK:
        return ""
    with streams_lock:
        entry = streams.get(sid)
    return tidy_stream_text(stream_worker.submit(recognizer.get_result, entry[0]).result()) if entry else ""


def stream_close(sid):
    with streams_lock:
        streams.pop(sid, None)


def speech_seconds(pcm_bytes):
    """Silero VAD：这段音频里实际有多少秒人声（铃声、音乐、噪音基本是 0）。约 10ms。"""
    if MOCK:
        return 1.0

    def work():
        a = np.frombuffer(pcm_bytes, dtype=np.float32)
        w = vad.config.silero_vad.window_size
        vad.reset()
        total = 0
        for i in range(0, len(a) - w + 1, w):
            vad.accept_waveform(a[i:i + w])
            while not vad.empty():
                total += len(vad.front.samples)
                vad.pop()
        vad.flush()
        while not vad.empty():
            total += len(vad.front.samples)
            vad.pop()
        return total / 16000

    return stream_worker.submit(work).result()


def looks_like_prompt(text, lang):
    """Whisper 把提示词原样吐出来（安静片段上常见）。"""
    from difflib import SequenceMatcher

    p = asr_prompt(lang)
    if not p or not text:
        return False
    a, b = re.sub(r"\W", "", text), re.sub(r"\W", "", p)
    # 提示词里带了热词，用户单独说一个热词（如"同声传译"）不能算泄漏：要占到提示词的四成以上才算
    return len(a) >= max(4, 0.4 * len(b)) and (a in b or SequenceMatcher(None, a, b).ratio() > 0.6)


def whisper_detect(pcm_bytes):
    """流式识别没给出结果时的兜底：让 Whisper 在中英文里二选一。"""
    import mlx.core as mx
    from mlx_whisper import audio as A
    from mlx_whisper.transcribe import ModelHolder

    m = ModelHolder.get_model(ASR_MODEL, mx.float16)
    mel = A.log_mel_spectrogram(np.frombuffer(pcm_bytes, dtype=np.float32), n_mels=m.dims.n_mels)
    seg = A.pad_or_trim(mel, A.N_FRAMES, axis=-2).astype(mx.float16)
    _, probs = m.detect_language(seg)
    return "zh" if probs.get("zh", 0) > probs.get("en", 0) else "en"


def english_voices():
    if MOCK:
        return [DEFAULT_VOICE]
    return [v for v in kokoro.get_voices() if v[:2] in ("af", "am", "bf", "bm")]


def synthesize(text, voice):
    """英文文本 → WAV 字节。"""
    import soundfile as sf

    lang = "en-us" if voice[0] == "a" else "en-gb"
    samples, sr = tts_worker.submit(kokoro.create, text, voice=voice, speed=1.0, lang=lang).result()
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


def warmup():
    """跑一次极小的识别+翻译，把模型权重留在内存里。"""
    if MOCK:
        return
    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=ASR_MODEL, language="en")
    list(stream_generate(model, tokenizer, prompt=build_prompt("hi", "en2zh", []), max_tokens=2))
    list(stream_generate(model, tokenizer, prompt=build_prompt("你好", "zh2en", []), max_tokens=2, logits_processors=[no_cjk]))


last_used = time.time()


def keep_warm():
    """内存紧张时（同时开录屏、浏览器），系统会把闲着的模型换到磁盘，下一句要卡十几秒。
    空闲时每 20 秒碰一下模型，让它一直留在内存里。"""
    while True:
        time.sleep(5)
        if time.time() - last_used > 20 and gpu.acquire(blocking=False):
            try:
                warmup()
            finally:
                gpu.release()
            synthesize("Ready.", DEFAULT_VOICE)
            globals()["last_used"] = time.time()


def transcribe(pcm_bytes, lang):
    if MOCK:
        return "(mock transcript)" if lang == "en" else "（模拟识别）"
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=ASR_MODEL,
        language=lang,
        initial_prompt=asr_prompt(lang),
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    # 按片段过滤：静音概率高 / 置信度很低 / 高度重复（压缩率高）的片段丢掉，常见幻觉句子整段删掉
    parts = []
    for seg in result.get("segments") or []:
        s = (seg.get("text") or "").strip()
        nsp, lp, cr = seg.get("no_speech_prob", 0), seg.get("avg_logprob", 0), seg.get("compression_ratio", 0)
        if not s or (nsp > 0.5 and lp < -0.5) or lp < -1.2 or cr > 2.4:
            continue
        s = HALLUCINATION_TAIL.sub("", s).strip()
        if s and not HALLUCINATIONS.match(s):
            parts.append(s)
    text = apply_fixes(("" if lang == "zh" else " ").join(parts).strip())
    if HALLUCINATIONS.match(text):
        return ""
    # Whisper 偶尔陷入重复循环（"计划计划计划……"），同一小段连续重复 4 次以上就只留一次
    text = re.sub(r"(.{1,8}?)\1{3,}", r"\1", text)
    # 冒出俄文/韩文/日文假名等文字 = 乱码；该说中文的一路一个汉字都没有 = 杂音/回声；和提示词几乎一样 = 提示词泄漏
    if FOREIGN_SCRIPT.search(text) or looks_like_prompt(text, lang):
        return ""
    if lang == "zh" and not re.search(r"[一-鿿]", text):
        return ""
    if lang == "zh":  # Whisper 中文常输出半角标点，换成全角
        text = text.translate(str.maketrans({",": "，", "?": "？", "!": "！", ":": "：", ";": "；"}))
    return text


# --- 草稿翻译：说话过程中译半句话；结尾很好猜时把预测的结尾放进 [方括号] ---
DRAFT_SYSTEM = (
    "You are a live interpreter in a technical job interview (English-speaking interviewer, "
    "Chinese-speaking candidate). You get an unfinished fragment of live speech recognition "
    "(no punctuation, may contain recognition errors such as words that sound alike). Translate it. "
    "If the ending is highly predictable, also translate the most likely ending and put that part "
    "inside [square brackets]; otherwise end with an ellipsis. Never answer the question. "
    "Output only the translation."
)
DRAFT_DIR = {
    "en2zh": "Translate this English fragment into Simplified Chinese. Output Chinese only.",
    "zh2en": "Translate this Chinese fragment into English. Output English only.",
}
DRAFT_SHOTS = [
    ("en2zh", "How would you design a rate limit", "你会如何设计一个限流[器？]"),
    ("en2zh", "Tell me about a time when you", "跟我讲讲你……的一次经历"),
    ("zh2en", "我之前在一家公司负责后端开", "I was previously responsible for back-end develop[ment at a company.]"),
    ("zh2en", "这个项目最大的难点是", "The biggest challenge of this project was……"),
    ("en2zh", "Can you walk me through how you would", "你能给我讲讲你会如何……"),
    ("zh2en", "我们用了一个月就把系统上", "It took us just one month to get the system [up and running.]"),
]

# --- 判断"这句话说完没有"：看模型认为下一个字是句号/问号的概率 ---
END_PREFIX = (
    "Transcript of a technical job interview between an English-speaking interviewer "
    "and a Chinese-speaking candidate.\n\n"
)
END_MARKS = ["。", "？", "！", ".", "?", "!", "\n", " .", " ?", "?\n", ".\n"]

# 固定不变的提示前缀预先算好（KV 缓存），每次只算新加的几个字
caches = {}


def _cached(prefix_ids):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    c = make_prompt_cache(model)
    model(mx.array([prefix_ids]), cache=c)
    mx.eval([x.state for x in c])
    return c


def _chat_prefix_cache(msgs):
    """把固定的 system + 示例对话算好存起来，之后每次只算新加的消息。"""
    probe = tokenizer.apply_chat_template(msgs + [{"role": "user", "content": "X"}], add_generation_prompt=True)
    pre = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
    k = 0
    while k < len(pre) and probe[k] == pre[k]:
        k += 1
    return (probe[:k], _cached(probe[:k]))


def build_caches():
    caches["draft"] = _chat_prefix_cache(draft_messages())
    caches["final"] = _chat_prefix_cache(final_messages())
    cached = _cached
    end_ids = tokenizer.encode(END_PREFIX)
    caches["end"] = (end_ids, cached(end_ids))
    caches["end_marks"] = sorted({tokenizer.encode(e, add_special_tokens=False)[0] for e in END_MARKS})


def final_messages():
    msgs = [{"role": "system", "content": system_prompt()}]
    for d, src, out in FINAL_SHOTS:
        msgs += [{"role": "user", "content": f"{DIRECTIONS[d]}\n{src}"}, {"role": "assistant", "content": out}]
    return msgs


def draft_messages():
    msgs = [{"role": "system", "content": DRAFT_SYSTEM}]
    for d, src, out in DRAFT_SHOTS:
        msgs += [{"role": "user", "content": f"{DRAFT_DIR[d]}\nFragment: {src}"}, {"role": "assistant", "content": out}]
    return msgs


def with_cache(name, ids):
    """返回 (剩下要算的 token, 一份缓存副本)；前缀对不上就不用缓存。"""
    import copy

    prefix, cache = caches[name]
    if ids[: len(prefix)] == prefix:
        return ids[len(prefix):], copy.deepcopy(cache)
    return ids, None


def end_probability(text, lang):
    """这段实时字幕后面紧跟句号/问号的概率：很低 = 肯定没说完。"""
    if MOCK:
        return 0.5
    import mlx.core as mx

    who = "Candidate" if lang == "zh" else "Interviewer"
    ids, cache = with_cache("end", tokenizer.encode(END_PREFIX + f"{who}: {text}"))
    logits = model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32)
    probs = mx.softmax(logits)
    return probs[mx.array(caches["end_marks"])].sum().item()


def build_prompt(text, direction, history, draft=False):
    if draft:
        msgs = draft_messages() + [{"role": "user", "content": f"{DRAFT_DIR[direction]}\nFragment: {text}"}]
    else:
        msgs = final_turns(text, direction, history)
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)


def final_turns(text, direction, history):
    """正式翻译时送给模型的完整对话（规则 + 示例 + 上下文 + 这一句）。纠错数据导出成 LoRA 训练集时也用它，保证训练和使用时提示词一样。"""
    # 上下文用"原文 → 译文"成对的历史，像模型自己翻译过的一样：用词前后一致，也不容易把上一句复述出来。
    # 很短的句子（≤4 个汉字或 ≤3 个英文词）不带上下文——以前 "English" 被译成了上一句的"你的麦克风没打开"。
    msgs = final_messages()
    short = len(re.findall(r"[\u4e00-\u9fff]", text)) <= 4 and len(re.findall(r"[A-Za-z]+", text)) <= 3
    if not short:
        for h in history[-4:]:
            if h.get("tr"):
                d = "zh2en" if h.get("who") == "me" else "en2zh"
                msgs += [{"role": "user", "content": f"{DIRECTIONS[d]}\n{h.get('text', '')}"},
                         {"role": "assistant", "content": h["tr"]}]
    msgs.append({"role": "user", "content": f"{DIRECTIONS[direction]}\n{text}"})
    return msgs


def stream_translation(text, direction, history, draft=False):
    if MOCK:
        for ch in ("（模拟翻译）" if direction == "en2zh" else "(mock) ") + text:
            time.sleep(0.02)
            yield ch
        return
    if not draft and "final" not in caches:  # 热词表的固定译法改过：重建翻译规则的缓存
        caches["final"] = _chat_prefix_cache(final_messages())
    prompt = build_prompt(text, direction, history, draft)
    prompt, cache = with_cache("draft" if draft else "final", prompt)
    out = ""
    procs = [no_cjk] if direction == "zh2en" else None
    for r in stream_generate(model, tokenizer, prompt=prompt, max_tokens=120 if draft else 400, prompt_cache=cache,
                             logits_processors=procs):
        if draft and finals_waiting > 0:
            raise Preempted()  # 有正式任务在等，草稿不译了
        if r.text:
            out += r.text
            if LEAK.search(out):  # 模型开始复述提示词/上下文：立刻停，不再往外送
                log(f"拦截到复述提示词，已停止：{out[:80]!r}")
                return
            yield r.text


class Preempted(Exception):
    pass


# 中译英时不许输出汉字：生成每个字时，把所有含汉字（或半个汉字字节）的候选直接排除。
# 以前"我们下周再对接一下"会稳定地译成 "We'll对接 next week. We'll connect next week."
_no_cjk_mask = None


def no_cjk(tokens, logits):
    global _no_cjk_mask
    import mlx.core as mx

    if _no_cjk_mask is None or _no_cjk_mask.shape[-1] != logits.shape[-1]:
        n = logits.shape[-1]
        bad = [i for i in range(min(n, len(tokenizer.vocab))) if re.search(r"[\u3400-\u9fff\ufffd]", tokenizer.decode([i]))]
        m = mx.zeros((n,), dtype=logits.dtype)
        m[mx.array(bad)] = -float("inf")
        _no_cjk_mask = m
        log(f"中译英禁用汉字：{len(bad)} 个候选")
    return logits + _no_cjk_mask


LEAK = re.compile(r"\[(Candidate|Interviewer)\]|(Candidate|Interviewer):|for context|Recent conversation|do not translate", re.I)


def free_gpu_cache():
    if not MOCK:
        import mlx.core as mx

        mx.clear_cache()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --- 会后纠错：每句正式翻译都记到 data/history/日期.jsonl；纠错页面改过的写到 data/corrections.jsonl（同一句以最后一次为准）---
DATA = HERE / "data"
CORRECTIONS = DATA / "corrections.jsonl"
_data_lock = threading.Lock()


def _append(path, obj):
    with _data_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def record(text, direction, history, result, first_ms, voice_id=""):
    t = time.time()
    _append(DATA / "history" / f"{time.strftime('%Y-%m-%d', time.localtime(t))}.jsonl", {
        "id": f"{int(t * 1000)}", "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
        "dir": direction, "src": text, "tr": result.strip(), "first_ms": first_ms, "scene": current_scene,
        "model": MT_MODEL.rsplit("/", 1)[-1] + ("+LoRA" if MT_ADAPTER else ""),
        "history": [{k: h.get(k, "") for k in ("who", "text", "tr")} for h in history[-4:]],
        **({"voice_id": voice_id} if voice_id else {})})


VOICE = DATA / "voice"


def save_voice(pcm_bytes, text, lang, speech_sec, source):
    """「保存我的录音」开着时：这句话的原始音频存成 16kHz 单声道 wav，文字和时长记在旁边。
    以后攒起来做声音克隆（英文用你自己的声音说）、识别微调、"像你一样说话"的训练。"""
    import wave

    import numpy as np

    t = time.time()
    vid = f"{int(t * 1000)}"
    day = time.strftime("%Y-%m-%d", time.localtime(t))
    path = VOICE / day / f"{vid}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.frombuffer(pcm_bytes, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((a * 32767).astype("<i2").tobytes())
    _append(VOICE / f"{day}.jsonl", {"id": vid, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
                                     "file": f"{day}/{vid}.wav", "text": text, "lang": lang, "sec": round(len(a) / 16000, 2),
                                     "speech_sec": round(speech_sec, 2), "source": source, "asr_model": Path(ASR_MODEL).name})
    return vid


# --- 读稿录音（/record）：照着 recording/script.txt 一句一句读，录音和正确文字直接配对，不用再校对 ---
# 录音稿：有 script.local.txt（自己的私人版本，不进 git）就用它，否则用公开的通用版 script.txt
SCRIPT_FILE = next(f for f in (HERE / "recording" / "script.local.txt", HERE / "recording" / "script.txt") if f.exists() or f.name == "script.txt")
REC = DATA / "recording"


def script_lines():
    """录音稿：[{id, section, text, free}]。id 由文字算出，稿子改了顺序也对得上以前的录音。"""
    import hashlib

    out, section = [], ""
    if SCRIPT_FILE.exists():
        for line in SCRIPT_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or (line.startswith("#") and not line.startswith("## ")):
                continue
            if line.startswith("## "):
                section = line[3:].strip()
                continue
            out.append({"id": hashlib.sha1(line.encode()).hexdigest()[:10], "section": section, "text": line,
                        "free": section.startswith("10") or "自由讲" in section})
    return out


def recording_takes():
    """每句最后一次录的那条（重录的以最新为准）。"""
    takes = {}
    for t in _read_jsonl(REC / "takes.jsonl"):
        if t.get("deleted"):
            takes.pop(t["line_id"], None)
        elif "file" in t:  # 新录的一条：整条替换（重录后以前改的文字作废）
            takes[t["line_id"]] = t
        elif t["line_id"] in takes:  # 对已有录音的修改（自由讲改文字）
            takes[t["line_id"]] = {**takes[t["line_id"]], **t}
    return takes


def wav_info(data):
    """读 wav：返回 (采样率, float32 单声道数组)。"""
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("只收 16 位 wav")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return sr, a


def to_16k(a, sr):
    import numpy as np

    if sr == 16000:
        return a.astype(np.float32)
    n = int(len(a) * 16000 / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def review_items():
    """所有记下来的句子（新的在前），附上纠错结果。"""
    fixed = {c["id"]: c for c in _read_jsonl(CORRECTIONS)}
    items = []
    for f in sorted((DATA / "history").glob("*.jsonl"), reverse=True) if (DATA / "history").is_dir() else []:
        for it in reversed(_read_jsonl(f)):
            if it.get("voice_id"):
                day = time.strftime("%Y-%m-%d", time.localtime(int(it["voice_id"]) / 1000))
                it["voice_url"] = f"/voice/{day}/{it['voice_id']}.wav"
            items.append({**it, "fix": fixed.get(it["id"])})
    return items


def hotword_suggestions(old, new):
    """原文改过的地方 -> 可以加进热词表的"错的写法 => 对的写法"（只取短的替换，长的多半是整句重说）。"""
    import difflib

    out = []
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if op == "replace":
            # 太短的往前多带两个字，避免"错一个字就全局替换"误伤别的词（页面上可以再改）
            k = min(a1, b1, 2) if a2 - a1 < 3 else 0
            a1, b1 = a1 - k, b1 - k
            # 英文按整词取（"Kindy" => "Kimi"，不要 "ndy" => "mi"）
            while a1 > 0 and b1 > 0 and old[a1 - 1].isascii() and old[a1 - 1].isalnum() and old[a1 - 1] == new[b1 - 1]:
                a1, b1 = a1 - 1, b1 - 1
            while a2 < len(old) and b2 < len(new) and old[a2].isascii() and old[a2].isalnum() and old[a2] == new[b2]:
                a2, b2 = a2 + 1, b2 + 1
            a, b = old[a1:a2], new[b1:b2]
            if 2 <= len(a) <= 12 and a != b and (a, b) not in hotwords()["fixes"]:
                out.append(f"{a} => {b}")
    return out


class _Held:
    """GPU 锁已经拿到了（草稿翻译用非阻塞方式抢到的），退出时释放。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        free_gpu_cache()
        gpu.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/review":
            body = (HERE / "review.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif re.fullmatch(r"/voice/\d{4}-\d{2}-\d{2}/\d+\.wav", self.path):
            f = VOICE / self.path[len("/voice/"):]
            if not f.exists():
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/record":
            body = (HERE / "record.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/recording/script":
            takes = recording_takes()
            lines = [{**ln, "take": takes.get(ln["id"])} for ln in script_lines()]
            self._json(200, {"lines": lines})
        elif re.fullmatch(r"/recording/[0-9a-f]{10}/\d+\.wav", self.path):
            f = REC / self.path[len("/recording/"):]
            if not f.exists():
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/review/items":
            self._json(200, {"items": review_items()})
        elif self.path == "/info":
            self._json(200, {"mt_model": MT_MODEL + ("+LoRA" if MT_ADAPTER else ""), "asr_model": ASR_MODEL, "scene": current_scene, "mock": MOCK})
        elif self.path == "/scenes":
            self._json(200, {"scenes": list_scenes(), "current": current_scene})
        elif self.path == "/voices":
            self._json(200, {"voices": english_voices(), "default": DEFAULT_VOICE})
        else:
            self.send_error(404)

    def do_POST(self):
        global last_used
        last_used = time.time()
        url = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if url.path == "/recording/save":
            # 读稿页面录完一句：body 是 wav（浏览器原始采样率，16 位单声道）
            q = parse_qs(url.query)
            lid = q.get("id", [""])[0]
            line = next((ln for ln in script_lines() if ln["id"] == lid), None)
            if not line:
                self.send_error(400)
                return
            try:
                sr, a = wav_info(body)
            except Exception as e:
                self._json(400, {"error": f"{type(e).__name__}: {e}"})
                return
            import numpy as np

            t = time.time()
            name = f"{int(t * 1000)}.wav"
            (REC / lid).mkdir(parents=True, exist_ok=True)
            (REC / lid / name).write_bytes(body)
            pcm16k = to_16k(a, sr)
            sp = speech_seconds(pcm16k.tobytes())
            take = {"line_id": lid, "section": line["section"], "text": line["text"], "free": line["free"],
                    "file": f"{lid}/{name}", "sr": sr, "sec": round(len(a) / sr, 2), "speech_sec": round(sp, 2),
                    "peak": round(float(np.abs(a).max()) if len(a) else 0, 3),
                    "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)) if len(a) else -99, 1),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))}
            if line["free"] and sp >= MIN_SPEECH_SEC and not MOCK:
                # 自由讲没有稿子：先识别出文字，页面上再改
                with FinalTask():
                    t0 = time.time()
                    take["asr_text"] = take["transcript"] = transcribe(pcm16k.tobytes(), "zh")
                log(f"自由讲识别 {take['sec']}s音频 {int((time.time() - t0) * 1000)}ms: {take['asr_text'][:60]}")
            _append(REC / "takes.jsonl", take)
            log(f"读稿录音 {lid} {take['sec']}s 人声{sp:.1f}s 峰值{take['peak']} 音量{take['rms_db']}dB: {line['text'][:30]}")
            self._json(200, {"take": take})
        elif url.path == "/recording/text":
            # 自由讲：改好的文字
            req = json.loads(body or b"{}")
            lid = req.get("id") or ""
            if lid not in recording_takes():
                self.send_error(404)
                return
            _append(REC / "takes.jsonl", {"line_id": lid, "transcript": (req.get("text") or "").strip(), "checked": True})
            self._json(200, {"saved": True})
        elif url.path == "/recording/delete":
            lid = json.loads(body or b"{}").get("id") or ""
            _append(REC / "takes.jsonl", {"line_id": lid, "deleted": True})
            self._json(200, {"deleted": True})
        elif url.path == "/correct":
            # 纠错页面提交：status = ok（译得对）/ fixed（改过）/ skip（杂音、没意义，不进评测）
            req = json.loads(body or b"{}")
            if not req.get("id") or req.get("status") not in ("ok", "fixed", "skip"):
                self.send_error(400)
                return
            fix = {k: (req.get(k) or "").strip() for k in ("id", "status", "src", "tr", "note")}
            fix["by"] = (req.get("by") or "you").strip()  # 谁标的：you = 你自己；claude = 大模型老师
            fix["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _append(CORRECTIONS, fix)
            old = (req.get("old_src") or "").strip()
            self._json(200, {"saved": True, "hotwords": hotword_suggestions(old, fix["src"]) if fix["src"] and old else []})
        elif url.path == "/hotword":
            line = (json.loads(body or b"{}").get("line") or "").strip()
            if not line or "\n" in line:
                self.send_error(400)
                return
            with _data_lock:
                with open(HOTWORDS_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            log(f"热词表新增：{line}")
            self._json(200, {"added": line})
        elif url.path == "/scene":
            global current_scene
            name = (json.loads(body or b"{}").get("name") or "").strip()
            if name and name not in list_scenes():
                self._json(404, {"error": f"没有这个场景：{name}"})
                return
            current_scene = name
            hotwords()  # 立刻重新读，日志里能看到切换结果
            if not MOCK and "final" not in caches:  # 切场景时就把新规则的缓存算好，免得切完第一句慢 3 秒
                with FinalTask():
                    caches["final"] = _chat_prefix_cache(final_messages())
            self._json(200, {"current": current_scene})
        elif url.path == "/warmup":
            with gpu:
                t = time.time()
                warmup()
            self._json(200, {"ms": int((time.time() - t) * 1000)})
        elif url.path == "/tts":
            req = json.loads(body or b"{}")
            text = (req.get("text") or "").strip()
            voice = req.get("voice") or DEFAULT_VOICE
            if not text or MOCK or voice not in english_voices():
                self.send_error(400)
                return
            try:
                t = time.time()
                wav = synthesize(text, voice)
                log(f"朗读 {voice} {int((time.time() - t) * 1000)}ms: {text}")
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        elif url.path == "/stream":
            # 说话过程中每 ~0.3 秒送一小段音频，返回实时字幕和判断出的语种（CPU，不占 GPU）
            sid = parse_qs(url.query).get("sid", [""])[0]
            try:
                text = stream_feed(sid, body)
                self._json(200, {"text": text, "lang": detect_lang(text)})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
        elif url.path == "/endcheck":
            # 说话中一出现停顿就来问：这句话说完了吗？（抢在草稿前面，但最多等 0.3 秒 GPU）
            req = json.loads(body or b"{}")
            global finals_waiting
            with finals_lock:
                finals_waiting += 1
            got = gpu.acquire(timeout=0.3)
            with finals_lock:
                finals_waiting -= 1
            if not got:
                self._json(409, {"busy": True})
                return
            try:
                t = time.time()
                p = end_probability(req.get("text") or "", req.get("lang") or "en")
                self._json(200, {"p": p, "ms": int((time.time() - t) * 1000)})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            finally:
                gpu.release()
        elif url.path == "/transcribe":
            q = parse_qs(url.query)
            lang = q.get("lang", ["auto"])[0]
            sid = q.get("sid", [""])[0]
            if lang not in ("auto", *ASR_PROMPT):
                self.send_error(400)
                return
            if lang == "auto":
                lang = detect_lang(stream_text(sid)) if sid else None
            stream_close(sid)
            try:
                # 先用 Silero VAD 看有没有人声（CPU，约 10ms）：铃声、音乐、噪音直接丢掉，不占 GPU
                sp = speech_seconds(body)
                if sp < MIN_SPEECH_SEC:
                    log(f"跳过[{sid or '-'}] {len(body) / 64000:.1f}s音频 只有{sp:.2f}s人声")
                    self._json(200, {"text": "", "lang": lang if lang != "auto" else "en", "ms": 0, "skipped": "no_speech"})
                    return
                with FinalTask():
                    t = time.time()
                    if lang is None and not MOCK:
                        lang = whisper_detect(body)
                    lang = lang or "en"
                    text = transcribe(body, lang)
                ms = int((time.time() - t) * 1000)
                log(f"识别[{sid or '-'}] {lang} {len(body) / 64000:.1f}s音频 人声{sp:.1f}s {ms}ms: {text}")
                vid = ""
                save = q.get("save", [""])[0]  # 页面上「保存我的录音」开着时带 save=me（会议模式我的麦克风）或 save=live（现场模式）
                if save in ("me", "live") and text:
                    vid = save_voice(body, text, lang, sp, save)
                self._json(200, {"text": text, "lang": lang, "ms": ms, "voice_id": vid})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
        elif url.path == "/translate":
            req = json.loads(body or b"{}")
            text = (req.get("text") or "").strip()
            direction = req.get("direction")
            draft = bool(req.get("draft"))
            if not text or direction not in DIRECTIONS:
                self.send_error(400)
                return
            # 草稿翻译：GPU 忙或有正式任务在等就不做
            if draft:
                if finals_waiting > 0 or not gpu.acquire(blocking=False):
                    self._json(409, {"busy": True})
                    return
                lock = _Held()
            else:
                lock = FinalTask()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(obj):
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            with lock:
                t = time.time()
                try:
                    first = None
                    result = ""
                    for chunk in stream_translation(text, direction, req.get("history") or [], draft):
                        if first is None:
                            first = int((time.time() - t) * 1000)
                        result += chunk
                        send({"t": chunk})
                    send({"done": True, "first_ms": first, "ms": int((time.time() - t) * 1000)})
                    if not draft:
                        if result.strip() and not req.get("no_record"):  # 评测脚本重译时不记
                            record(text, direction, req.get("history") or [], result, first, req.get("voice_id") or "")
                        log(f"翻译 {direction} 首字{first}ms 共{int((time.time() - t) * 1000)}ms: {text} => {result}")
                except Preempted:
                    send({"preempted": True})
                except BrokenPipeError:
                    pass
                except Exception as e:
                    try:
                        send({"error": f"{type(e).__name__}: {e}"})
                    except BrokenPipeError:
                        pass
        else:
            self.send_error(404)


if __name__ == "__main__":
    if not MOCK:
        print("正在加载本地模型……", flush=True)
        load_models()
        threading.Thread(target=keep_warm, daemon=True).start()
    print(f"同声传译演示: http://localhost:{PORT}  ({'模拟模式' if MOCK else '全本地'})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
