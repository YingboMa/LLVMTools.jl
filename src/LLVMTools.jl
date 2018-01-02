__precompile__()

module LLVMTools
@reexport using SIMD
import SIMD:llvmins, llvmwrap, llvmconst, llvmtype, suffix
export va_start, va_end, va_copy, prefetch, pcmarker, bitreverse, ctlz,
       cttz, gcroot, gcread, gcwrite, returnaddress, frameaddress, stacksave,
       stackrestore, readcyclecounter, clear_cache, bitreverse, ctpop

# TODO: Clean up with e.g. ccall("llvm.clear_cache", llvmcall,
#                                Nothing, (Ptr{Nothing}, Ptr{Nothing},
#                                x, y)

# Variable Argument Handling Intrinsics
# https://llvm.org/docs/LangRef.html#variable-argument-handling-intrinsics
@inline function va_start(arglist::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.va_start(i8*)",
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.va_start(i8* %ptr)
                   ret void
                   """),
                  Nothing, Tuple{UInt64},
                  UInt64(arglist))
end

@inline function va_end(arglist::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.va_end(i8*)",
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.va_end(i8* %ptr)
                   ret void
                   """),
                  Nothing, Tuple{UInt64},
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
                  Nothing, Tuple{UInt64, UInt64},
                  UInt64(destarglist), UInt64(srcarglist))
end

# TODO: Accurate Garbage Collection Intrinsics
# https://llvm.org/docs/LangRef.html#accurate-garbage-collection-intrinsics

@inline function gcroot(ptrloc::Ptr{Ptr{S}}, metadata::Ptr{T}) where {S,T}
    Base.llvmcall(("declare void @llvm.gcroot(i8**, i8*)",
                   """
                   %ptr1 = inttoptr i64 %0 to i8**
                   %ptr2 = inttoptr i64 %1 to i8*
                   call void @llvm.gcroot(i8** %ptr1, i8* %ptr2)
                   ret void
                   """),
                  Nothing, Tuple{UInt64, UInt64},
                  UInt64(ptrloc), UInt64(metadata))
end

@inline function gcread(objptr::Ptr{S}, ptr::Ref{Ptr{T}}) where {S,T}
    Base.llvmcall(("declare void @llvm.gcread(i8*, i8**)",
                   """
                   %ptr1 = inttoptr i64 %0 to i8*
                   %ptr2 = inttoptr i64 %1 to i8**
                   %ptr = call i8* @llvm.gcread(i8* %ptr1, i8** %ptr2)
                   %3 = ptrtoint i8* %ptr to i64
                   ret i64 %3
                   """),
                  UInt64, Tuple{UInt64, UInt64},
                  UInt64(objptr), UInt64(ptr)) |> Ptr{Nothing}
end

@inline function gcwrite(p1::Ptr{T}, obj::Ptr{S}, p2::Ref{Ptr{Z}}) where {S,T,Z}
    Base.llvmcall(("declare void @llvm.gcwrite(i8*, i8*, i8**)",
                   """
                   %ptr1 = inttoptr i64 %0 to i8*
                   %ptr2 = inttoptr i64 %1 to i8*
                   %ptr3 = inttoptr i64 %2 to i8**
                   call void @llvm.gcwrite(i8* %ptr1, i8* %ptr2, i8** %ptr3)
                   ret void
                   """),
                  Nothing, Tuple{UInt64, UInt64, UInt64},
                  UInt64(p1), UInt64(obj), UInt64(p2))
end

# TODO: Code Generator Intrinsics
# https://llvm.org/docs/LangRef.html#id1374

@inline function returnaddress(::Type{Val{level}}) where level
    Base.llvmcall(("declare i8* @llvm.returnaddress(i32)",
                   """
                   %2 = call i8* @llvm.returnaddress(i32 %0)
                   %3 = ptrtoint i8* %2 to i64
                   ret i64 %3
                   """),
                   UInt64, Tuple{UInt32,},
                   UInt32(level)) |> Ptr{Nothing}
end

# LLVM ERROR: Program used external function 'llvm.addressofreturnaddress' which could not be resolved!
#@inline function addressofreturnaddress()
#    Base.llvmcall(("declare i8* @llvm.addressofreturnaddress()",
#                   """
#                   %1 = call i8* @llvm.addressofreturnaddress()
#                   %addr = ptrtoint i8* %1 to i64
#                   ret i64 %addr
#                   """),
#                  UInt64, Nothing) |> Ptr{Nothing}
#end

@inline function frameaddress(::Type{Val{level}}) where level
    Base.llvmcall(("declare i8* @llvm.frameaddress(i32)",
                   """
                   %2 = call i8* @llvm.frameaddress(i32 %0)
                   %3 = ptrtoint i8* %2 to i64
                   ret i64 %3
                   """),
                   UInt64, Tuple{UInt32,},
                   UInt32(level)) |> Ptr{Nothing}
end

# TODO: ‘llvm.localescape’ and ‘llvm.localrecover’ Intrinsics
# TODO: ‘llvm.read_register’ and ‘llvm.write_register’ Intrinsics

@inline function stacksave()
    Base.llvmcall(("declare i8* @llvm.stacksave()",
                   """
                   %1 = call i8* @llvm.stacksave()
                   %2 = ptrtoint i8* %1 to i64
                   ret i64 %2
                   """),
                  UInt64, Nothing) |> Ptr{Nothing}
end

@inline function stackrestore(ptr::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.stackrestore(i8*)",
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.stackrestore(i8* %ptr)
                   ret void
                   """),
                  Nothing, Tuple{UInt64},
                  UInt64(ptr))
end

#TODO: ‘llvm.get.dynamic.area.offset’ Intrinsic

@inline function prefetch(address::Ptr{T}, ::Type{Val{rw}}, ::Type{Val{locality}}, ::Type{Val{cachetype}}) where T
    Base.llvmcall((""" declare void @llvm.prefetch(i8*, i32, i32, i32) """,
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.prefetch(i8* %ptr, i32 %1, i32 %2, i32 %3)
                   ret void
                   """),
                  Nothing, Tuple{UInt64, Int32, Int32, Int32},
                  UInt64(address), Int32(rw), Int32(locality), Int32(cachetype))
end

@inline function pcmarker(id::Integer)
    Base.llvmcall((""" declare void @llvm.pcmarker(i32) """,
                   """
                   call void @llvm.pcmarker(i32 %0)
                   ret void
                   """),
                  Nothing, Tuple{Int32},
                  Int32(id))
end

@inline function readcyclecounter()
    Base.llvmcall(("declare i64 @llvm.readcyclecounter()",
                   """
                   %1 = call i64 @llvm.readcyclecounter()
                   ret i64 %1
                   """),
                  UInt64, Nothing)
end

@inline function clear_cache(p1::Ptr{T}, p2::Ptr{T}) where T
    Base.llvmcall(("declare void @llvm.clear_cache(i8*, i8*)",
                   """
                   %ptr1 = inttoptr i64 %0 to i8*
                   %ptr2 = inttoptr i64 %1 to i8*
                   call void @llvm.clear_cache(i8* %ptr1, i8* %ptr2)
                   ret void
                   """),
                  Nothing, Tuple{UInt64, UInt64},
                  UInt64(p1), UInt64(p1))
end

# TODO: Standard C Library Intrinsics
# https://llvm.org/docs/LangRef.html#standard-c-library-intrinsics

# TODO: Bit Manipulation Intrinsics
# https://llvm.org/docs/LangRef.html#bit-manipulation-intrinsics
@generated function bitreverse(id::T) where T<:Union{Int16, UInt16, Int32, UInt32, Int64, UInt64}
    typ = llvmtype(T)
    dec = "declare $typ @llvm.bitreverse.$typ($typ)"
    instrs = String[]
    push!(instrs, "%2 = call $typ @llvm.bitreverse.$typ($typ %0)")
    push!(instrs, "ret $typ %2")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall(($dec, $(join(instrs, "\n"))),
                       $T, Tuple{$T}, id)
    end
end

# Already in base Julia
#@generated function lbswap(id::T) where T<:Union{Int16, UInt16, Int32, UInt32, Int64, UInt64}
#    typ = llvmtype(T)
#    dec = "declare $typ @llvm.bswap.$typ($typ)"
#    instrs = String[]
#    push!(instrs, "%2 = call $typ @llvm.bswap.$typ($typ %0)")
#    push!(instrs, "ret $typ %2")
#    quote
#        $(Expr(:meta, :inline))
#        Base.llvmcall(($dec, $(join(instrs, "\n"))),
#                       $T, Tuple{$T}, id)
#    end
#end

@generated function ctpop(id::T) where T<:Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,UInt128,Int128}
    typ = llvmtype(T)
    dec = "declare $typ @llvm.ctpop.$typ($typ)"
    instrs = String[]
    push!(instrs, "%2 = call $typ @llvm.ctpop.$typ($typ %0)")
    push!(instrs, "ret $typ %2")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall(($dec, $(join(instrs, "\n"))),
                       $T, Tuple{$T}, id)
    end
end

const ctlz = Core.Intrinsics.ctlz_int
const cttz = Core.Intrinsics.cttz_int

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
