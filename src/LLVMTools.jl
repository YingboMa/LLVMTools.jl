__precompile__()

module LLVMTools
@reexport using SIMD
import SIMD:llvmins, llvmwrap, llvmconst, llvmtype, suffix
export va_start, va_end, va_copy, prefetch_r, prefetch_w, pcmarker

# Variable Argument Handling Intrinsics
# https://llvm.org/docs/LangRef.html#variable-argument-handling-intrinsics
@inline function va_start(arglist::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.va_start(i8*)",
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.va_start(i8* %ptr)
                   ret void
                   """),
                  Void, Tuple{UInt64},
                  UInt64(arglist))
end

@inline function va_end(arglist::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.va_end(i8*)",
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.va_end(i8* %ptr)
                   ret void
                   """),
                  Void, Tuple{UInt64},
                  UInt64(arglist))
end

@inline function va_end(destarglist::Ptr{T}, srcarglist::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.va_copy(i8*)",
                   """
                   %ptr1 = inttoptr i64 %0 to i8*
                   %ptr2 = inttoptr i64 %1 to i8*
                   call void @llvm.va_end(i8* %ptr1, i8* %ptr2)
                   ret void
                   """),
                  Void, Tuple{UInt64, UInt64},
                  UInt64(destarglist), UInt64(srcarglist))
end

# TODO: Accurate Garbage Collection Intrinsics
# https://llvm.org/docs/LangRef.html#accurate-garbage-collection-intrinsics



# TODO: Code Generator Intrinsics
# https://llvm.org/docs/LangRef.html#id1374

@inline function _prefetch(address::Ptr{T}, rw::Integer, locality::Integer, cachetype::Integer) where T
    Base.llvmcall((""" declare void @llvm.prefetch(i8*, i32, i32, i32) """,
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.prefetch(i8* %ptr, i32 %1, i32 %2, i32 %3)
                   ret void
                   """),
                  Void, Tuple{UInt64, Int32, Int32, Int32},
                  UInt64(address), Int32(rw), Int32(locality), Int32(cachetype))
end

@inline prefetch_r(ptr::Ptr{T}) where T = _prefetch(ptr, 0, 3, 1)
@inline prefetch_w(ptr::Ptr{T}) where T = _prefetch(ptr, 1, 3, 1)

@inline function pcmarker(id::Integer)
    Base.llvmcall((""" declare void @llvm.pcmarker(i32) """,
                   """
                   call void @llvm.pcmarker(i32 %0)
                   ret void
                   """),
                  Void, Tuple{Int32},
                  Int32(id))
end


# TODO: Standard C Library Intrinsics
# https://llvm.org/docs/LangRef.html#standard-c-library-intrinsics

# TODO: Bit Manipulation Intrinsics
# https://llvm.org/docs/LangRef.html#bit-manipulation-intrinsics

# TODO: Arithmetic with Overflow Intrinsics
# https://llvm.org/docs/LangRef.html#arithmetic-with-overflow-intrinsics

# TODO: Specialised Arithmetic Intrinsics
# https://llvm.org/docs/LangRef.html#id1629

# TODO: Experimental Vector Reduction Intrinsics
# https://llvm.org/docs/LangRef.html#experimental-vector-reduction-intrinsics

# TODO: Half Precision Floating Point Intrinsics
# https://llvm.org/docs/LangRef.html#half-precision-floating-point-intrinsics

# TODO: ...
# https://llvm.org/docs/LangRef.html

end
