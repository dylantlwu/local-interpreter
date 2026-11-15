#!/bin/bash
# 从 BlackHole 官方开源代码（GPL-3.0，https://github.com/ExistentialAudio/BlackHole）在本机编译两个虚拟声卡：
#   BlackHole 2ch  —— 会议软件的扬声器选它，同传页面从这里听对方
#   BlackHole 16ch —— 会议软件的麦克风选它，同传页面把英文语音送到这里
# 只需要 Xcode 命令行工具（xcode-select --install），不需要完整的 Xcode。
# 用法：bash drivers/build_blackhole.sh   然后   sudo bash drivers/install_blackhole.sh
set -e
cd "$(dirname "$0")"
VERSION=v0.7.1
[ -d src ] || git clone -q --depth 1 --branch $VERSION https://github.com/ExistentialAudio/BlackHole.git src

for ch in 2 16; do
  name=BlackHole${ch}ch
  bid=audio.existential.$name
  out=build/$name.driver/Contents
  rm -rf build/$name.driver
  mkdir -p $out/MacOS $out/Resources
  clang -O2 -bundle -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 -DDEBUG=0 \
    -DkNumber_Of_Channels=$ch -DkPlugIn_BundleID="\"$bid\"" -DkDriver_Name="\"BlackHole\"" \
    -framework CoreAudio -framework CoreFoundation -framework Accelerate \
    -Wno-everything -o $out/MacOS/BlackHole src/BlackHole/BlackHole.c
  sed -e 's/${EXECUTABLE_NAME}/BlackHole/' \
      -e "s/\$(PRODUCT_BUNDLE_IDENTIFIER)/$bid/" \
      -e "s/\${PRODUCT_NAME}/$name/" \
      -e 's/$(MARKETING_VERSION)/0.7.1/' \
      -e "s/e395c745-4eea-4d94-bb92-46224221047c/$(uuidgen)/g" \
      src/BlackHole/BlackHole.plist > $out/Info.plist
  cp src/BlackHole/BlackHole.icns $out/Resources/
  codesign --force --deep --sign - build/$name.driver   # 本机 ad-hoc 签名
  echo "已编译 $name"
done
echo "下一步：sudo bash drivers/install_blackhole.sh"
