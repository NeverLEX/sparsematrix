
NDKROOT=$1
BLASROOT=$2
SYSROOT=$NDKROOT/sysroot
make clean
# Set LDFLAGS so that the linker finds the appropriate libgcc
export LDFLAGS="-L${NDKROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/lib/gcc/arm-linux-androideabi/4.9.x -lgcc"

# Set the clang cross compile flags
export CLANG_FLAGS="-target armv7-linux-androideabi -D__ANDROID_API__=16 -marm -mfpu=vfp -mfloat-abi=softfp \
  --sysroot=$NDKROOT/platforms/android-16/arch-arm -isystem $SYSROOT/usr/include/arm-linux-androideabi \
  -isystem $SYSROOT/usr/include -ffunction-sections -fdata-sections \
  -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward \
  --gcc-toolchain=$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64"

./configure --blas-root=$BLASROOT \
			--ar="$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/arm-linux-androideabi/bin/ar" \
			--as="$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/arm-linux-androideabi/bin/as" \
			--runlib="$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/arm-linux-androideabi/bin/ranlib" \
			--cxx="$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++" \
			--cxxflags=" -std=c++11 -fPIC -O2 ${CLANG_FLAGS}"

make mobile

GNUSTL=$NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/libgnustl_static.a
CXXFLAGS="--std=c++0x -static-libstdc++ -I./src/sparse -I./src/test -I$BLASROOT/include -I$NDKROOT/sysroot/usr/include -I$NDKROOT/sysroot/usr/include/arm-linux-androideabi/ -marm -march=armv7-a -mfpu=neon -mfloat-abi=softfp -O2 -Wall -Wextra -pedantic -latomic -llog -fPIC -pie -fPIE -frtti -D__ANDROID_API__=16 --sysroot=$NDKROOT/platforms/android-16/arch-arm --target=armv7-none-linux-androideabi --gcc-toolchain=$NDKROOT/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64 -isystem $SYSROOT/usr/include/arm-linux-androideabi -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include/ -isystem $NDKROOT/sources/cxx-stl/gnu-libstdc++/4.9/include/backward"
$NDKROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ $CXXFLAGS ./src/test/blas_test.cc -o ./src/test/blas_test ${BLASROOT}/lib/libopenblas.a ./src/sparse/libsparsematrix.a $GNUSTL

        
