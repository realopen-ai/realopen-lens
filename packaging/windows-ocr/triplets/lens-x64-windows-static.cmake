# Portable OCR: no separately installed VC runtime or third-party DLLs.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_BUILD_TYPE release)
if(PORT STREQUAL "tesseract")
    # OCR concurrency is bounded at the Lens worker level. Avoid a separate
    # LLVM/MSVC OpenMP runtime DLL and builder-specific CPU requirements.
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        -DOPENMP_BUILD=OFF -DWIN32_MT_BUILD=ON -DENABLE_NATIVE=OFF)
endif()
