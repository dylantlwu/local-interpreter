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

PORT = int(os.environ.get("PORT", "8765"))
MOCK = os.environ.get("MOCK") == "1"
ASR_MODEL = os.environ.get("ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
MT_MODEL = os.environ.get("MT_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
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

SYSTEM = (
    "You are a live interpreter in a technical job interview between an English-speaking "
    "interviewer and a Chinese-speaking candidate. Translate the given utterance faithfully "
    "and naturally. Keep technical terms (API, React, PostgreSQL, CI/CD, etc.) in their usual "
    "form. Speech-recognition text may contain errors; infer the intended meaning from context. "
    "Output only the translation, with no notes, quotes, or explanations."
)
DIRECTIONS = {
    "en2zh": "Translate this English into Simplified Chinese:",
    "zh2en": "Translate this Chinese into spoken, fluent English:",
}
# 给 Whisper 的提示：中文偏向简体、带标点
ASR_PROMPT = {
    "zh": "以下是普通话的句子，使用简体中文。我们在聊AI、Python、React、PostgreSQL、API和系统设计。",
    "en": "A technical interview about AI, Python, React, TypeScript, PostgreSQL, APIs, and system design.",
}
# Whisper 在噪音上常见的幻觉输出
HALLUCINATIONS = re.compile(
    r"^(thank you\.?|thanks for watching!?|you\.?|bye\.?|字幕.*|请不吝点赞.*|谢谢观看.*|\.+)$", re.I
)
HALLUCINATION_TAIL = re.compile(
    r"(请不吝点赞.*|字幕由.*|字幕志愿者.*|谢谢观看.*|明镜与点点.*|thanks for watching.*)$", re.I
)

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
    model, tokenizer = load(MT_MODEL)
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
    return text


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
        initial_prompt=ASR_PROMPT[lang],
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    # 按片段过滤：静音概率高的片段丢掉，常见幻觉句子（"请不吝点赞订阅……"之类）整段删掉
    parts = []
    for seg in result.get("segments") or []:
        s = (seg.get("text") or "").strip()
        if not s or seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -0.7:
            continue
        s = HALLUCINATION_TAIL.sub("", s).strip()
        if s and not HALLUCINATIONS.match(s):
            parts.append(s)
    text = ("" if lang == "zh" else " ").join(parts).strip()
    if HALLUCINATIONS.match(text):
        return ""
    # Whisper 偶尔陷入重复循环（"计划计划计划……"），同一小段连续重复 4 次以上就只留一次
    text = re.sub(r"(.{1,8}?)\1{3,}", r"\1", text)
    # 该说中文的一路：一个汉字都没有、或者冒出日文假名，基本是杂音/回声/幻觉，丢掉
    if lang == "zh" and (not re.search(r"[一-鿿]", text) or re.search(r"[぀-ヿ]", text)):
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


def build_caches():
    import copy  # noqa: F401
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    def cached(prefix_ids):
        c = make_prompt_cache(model)
        model(mx.array([prefix_ids]), cache=c)
        mx.eval([x.state for x in c])
        return c

    msgs = draft_messages()
    probe = tokenizer.apply_chat_template(msgs + [{"role": "user", "content": "X"}], add_generation_prompt=True)
    pre = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
    k = 0
    while k < len(pre) and probe[k] == pre[k]:
        k += 1
    caches["draft"] = (probe[:k], cached(probe[:k]))
    end_ids = tokenizer.encode(END_PREFIX)
    caches["end"] = (end_ids, cached(end_ids))
    caches["end_marks"] = sorted({tokenizer.encode(e, add_special_tokens=False)[0] for e in END_MARKS})


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
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    # 上下文放进 system 里（以前放在 user 里，遇到无意义的输入时模型会把上下文原样吐出来，还被朗读出去）
    context = "\n".join(
        f"- {'Interviewer' if h.get('who') == 'them' else 'Candidate'}: {h.get('text', '')}"
        for h in history[-6:]
    )
    system = SYSTEM
    if context:
        system += f"\n\nRecent conversation, for your understanding only. Never repeat or translate it:\n{context}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"{DIRECTIONS[direction]}\n{text}"}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def stream_translation(text, direction, history, draft=False):
    if MOCK:
        for ch in ("（模拟翻译）" if direction == "en2zh" else "(mock) ") + text:
            time.sleep(0.02)
            yield ch
        return
    prompt = build_prompt(text, direction, history, draft)
    cache = None
    if draft:
        prompt, cache = with_cache("draft", prompt)
    out = ""
    for r in stream_generate(model, tokenizer, prompt=prompt, max_tokens=120 if draft else 400, prompt_cache=cache):
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


LEAK = re.compile(r"\[(Candidate|Interviewer)\]|(Candidate|Interviewer):|for context|Recent conversation|do not translate", re.I)


def free_gpu_cache():
    if not MOCK:
        import mlx.core as mx

        mx.clear_cache()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


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
        elif self.path == "/voices":
            self._json(200, {"voices": english_voices(), "default": DEFAULT_VOICE})
        else:
            self.send_error(404)

    def do_POST(self):
        global last_used
        last_used = time.time()
        url = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if url.path == "/warmup":
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
                with FinalTask():
                    t = time.time()
                    if lang is None and not MOCK:
                        lang = whisper_detect(body)
                    lang = lang or "en"
                    text = transcribe(body, lang)
                ms = int((time.time() - t) * 1000)
                log(f"识别[{sid or '-'}] {lang} {len(body) / 64000:.1f}s音频 {ms}ms: {text}")
                self._json(200, {"text": text, "lang": lang, "ms": ms})
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
                    for chunk in stream_translation(text, direction, req.get("history") or [], draft):
                        if first is None:
                            first = int((time.time() - t) * 1000)
                        send({"t": chunk})
                    send({"done": True, "first_ms": first, "ms": int((time.time() - t) * 1000)})
                    if not draft:
                        log(f"翻译 {direction} 首字{first}ms 共{int((time.time() - t) * 1000)}ms")
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
