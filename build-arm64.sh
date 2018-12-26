
NDKROOT=$1
BLASROOT=$2
SYSROOT=$NDKROOT/sysroot
make clean
# Set LDFLAGS so that the linker finds the appropriate libgcc
export LDFLAGS="-L${NDKROOT}/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/lib/gcc/aarch64-linux-android/4.9.x -lm"

# Set the clang cross compile flags
export CLANG_FLAGS="-target aarch64-linux-android -D__ANDROID_API__=23 \
  --sysroot=$NDKROOT/platforms/android-23/arch-arm64 -isystem $SYSROOT/usr/include/aarch64-linux-android \
  -isystem $SYSROOT/usr/include -ffunction-sections -fdata-sections \
  -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward \
  --gcc-toolchain=$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64"

./configure --blas-root=$BLASROOT \
			--ar="$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/aarch64-linux-android/bin/ar" \
			--as="$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/aarch64-linux-android/bin/as" \
			--runlib="$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/aarch64-linux-android/bin/ranlib" \
			--cxx="$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++" \
			--cxxflags=" -std=c++11 -fPIC -O2 ${CLANG_FLAGS} -march=armv8-a+crc+crypto+fp+simd -mtune=cortex-a57" \
			--target="ARM64"

make mobile

GNUSTL=$NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/libgnustl_static.a
CXXFLAGS="--std=c++0x -static-libstdc++ -I./src/sparse -I./src/test -I$BLASROOT/include -I$NDKROOT/sysroot/usr/include -I$NDKROOT/sysroot/usr/include/aarch64-linux-android/ -march=armv8-a -ftree-vectorize -O2 -Wall -Wextra -pedantic -latomic -llog -fPIC -pie -fPIE -frtti -D__ANDROID_API__=23 --sysroot=$NDKROOT/platforms/android-23/arch-arm64 --target=aarch64-linux-android --gcc-toolchain=$NDKROOT/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64 -isystem $SYSROOT/usr/include/aarch64-linux-android -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward"
$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ $CXXFLAGS ./src/test/blas_test.cc -o ./src/test/blas_test ${BLASROOT}/lib/libopenblas.a ./src/sparse/libsparsematrix.a $GNUSTL
