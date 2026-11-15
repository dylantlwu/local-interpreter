#!/bin/bash
# 安装两个虚拟声卡（由 build_blackhole.sh 从 BlackHole v0.7.1 开源代码本机编译）：
#   BlackHole 2ch  —— 会议软件的"扬声器"选它，同传页面从这里听对方
#   BlackHole 16ch —— 会议软件的"麦克风"选它，同传页面把英文语音送到这里
# 用法：sudo bash drivers/install_blackhole.sh（先运行 bash drivers/build_blackhole.sh）
set -e
if [ "$(id -u)" != "0" ]; then echo "请用 sudo 运行"; exit 1; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST=/Library/Audio/Plug-Ins/HAL
for d in BlackHole2ch BlackHole16ch; do
  rm -rf "$DEST/$d.driver"
  cp -R "$HERE/build/$d.driver" "$DEST/"
  chown -R root:wheel "$DEST/$d.driver"
  chmod -R 755 "$DEST/$d.driver"
  echo "已安装 $d"
done
killall coreaudiod || true   # 重启系统音频服务，新声卡才会出现（声音会断一两秒）
sleep 2
system_profiler SPAudioDataType | grep -E "BlackHole" || echo "没看到 BlackHole，请重启电脑后再看"
