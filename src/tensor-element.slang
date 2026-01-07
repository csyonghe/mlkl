implementing mlkl;

// ============================================================================
// Core Interfaces for Generic Element Type Support
// ============================================================================

/// Interface for unpacked/computational element types.
/// This is the type used for arithmetic operations in shader code.
interface IUnpackedElement
{
    // Construction
    __init(int x);
    __init(float x);
    
    // Conversions (for mixed precision and constants)
    float toFloat();
    int toInt();
    static This fromFloat(float x);
    static This fromInt(int x);
    
    // Arithmetic
    This add(This rhs);
    This sub(This rhs);
    This mul(This rhs);
    This div(This rhs);
    This neg();
    
    // Comparison (return bool)
    bool lt(This rhs);   // <
    bool le(This rhs);   // <=
    bool gt(This rhs);   // >
    bool ge(This rhs);   // >=
    bool eq(This rhs);   // ==
    bool ne(This rhs);   // !=
    
    // Math functions
    This sin();
    This cos();
    This exp();
    This log();
    This abs();
    This sqrt();
    This rsqrt();
    This floor();
    This ceil();
    This tanh();
    This pow(This e);
    
    // Static functions
    static This max(This x, This y);
    static This min(This x, This y);
    
    // Wave intrinsics
    This waveActiveSum();
}

/// Interface for tensor storage element types.
/// Maps a storage type to its computational (unpacked) type.
interface ITensorElement
{
    associatedtype UnpackedType : IUnpackedElement;
    
    /// Unpack from storage format to computational format.
    UnpackedType unpack();
    
    /// Pack from computational format back to storage format.
    static This pack(UnpackedType value);
}

// ============================================================================
// Operator Overloads (Global Scope)
// ============================================================================

// Arithmetic operators
[ForceInline]
T operator+<T : IUnpackedElement>(T a, T b) { return a.add(b); }

[ForceInline]
T operator-<T : IUnpackedElement>(T a, T b) { return a.sub(b); }

[ForceInline]
T operator*<T : IUnpackedElement>(T a, T b) { return a.mul(b); }

[ForceInline]
T operator/<T : IUnpackedElement>(T a, T b) { return a.div(b); }

[ForceInline]
__prefix
T operator-<T : IUnpackedElement>(T a) { return a.neg(); }

// Comparison operators
[ForceInline]
bool operator< <T : IUnpackedElement>(T a, T b) { return a.lt(b); }

[ForceInline]
bool operator<= <T : IUnpackedElement>(T a, T b) { return a.le(b); }

[ForceInline]
bool operator> <T : IUnpackedElement>(T a, T b) { return a.gt(b); }

[ForceInline]
bool operator>= <T : IUnpackedElement>(T a, T b) { return a.ge(b); }

[ForceInline]
bool operator== <T : IUnpackedElement>(T a, T b) { return a.eq(b); }

[ForceInline]
bool operator!= <T : IUnpackedElement>(T a, T b) { return a.ne(b); }

// ============================================================================
// Extension: float
// ============================================================================

extension float : IUnpackedElement
{
    [ForceInline] __init(int x) { this = float(x); }
    
    [ForceInline] float toFloat() { return this; }
    [ForceInline] int toInt() { return int(this); }
    [ForceInline] static This fromFloat(float x) { return x; }
    [ForceInline] static This fromInt(int x) { return float(x); }
    
    [ForceInline] This add(This rhs) { return this + rhs; }
    [ForceInline] This sub(This rhs) { return this - rhs; }
    [ForceInline] This mul(This rhs) { return this * rhs; }
    [ForceInline] This div(This rhs) { return this / rhs; }
    [ForceInline] This neg() { return -this; }
    
    [ForceInline] bool lt(This rhs) { return this < rhs; }
    [ForceInline] bool le(This rhs) { return this <= rhs; }
    [ForceInline] bool gt(This rhs) { return this > rhs; }
    [ForceInline] bool ge(This rhs) { return this >= rhs; }
    [ForceInline] bool eq(This rhs) { return this == rhs; }
    [ForceInline] bool ne(This rhs) { return this != rhs; }
    
    [ForceInline] This sin() { return ::sin(this); }
    [ForceInline] This cos() { return ::cos(this); }
    [ForceInline] This exp() { return ::exp(this); }
    [ForceInline] This log() { return ::log(this); }
    [ForceInline] This abs() { return ::abs(this); }
    [ForceInline] This sqrt() { return ::sqrt(this); }
    [ForceInline] This rsqrt() { return ::rsqrt(this); }
    [ForceInline] This floor() { return ::floor(this); }
    [ForceInline] This ceil() { return ::ceil(this); }
    [ForceInline] This tanh() { return ::tanh(this); }
    [ForceInline] This pow(This e) { return ::pow(this, e); }
    
    [ForceInline] static This max(This x, This y) { return ::max(x, y); }
    [ForceInline] static This min(This x, This y) { return ::min(x, y); }
    
    [ForceInline] This waveActiveSum() { return WaveActiveSum(this); }
}

extension float : ITensorElement
{
    typealias UnpackedType = float;
    
    [ForceInline] UnpackedType unpack() { return this; }
    [ForceInline] static This pack(UnpackedType value) { return value; }
}

// ============================================================================
// Extension: half (float16)
// ============================================================================

extension half : IUnpackedElement
{
    [ForceInline] __init(int x) { this = half(x); }
    
    [ForceInline] float toFloat() { return float(this); }
    [ForceInline] int toInt() { return int(this); }
    [ForceInline] static This fromFloat(float x) { return half(x); }
    [ForceInline] static This fromInt(int x) { return half(x); }
    
    [ForceInline] This add(This rhs) { return this + rhs; }
    [ForceInline] This sub(This rhs) { return this - rhs; }
    [ForceInline] This mul(This rhs) { return this * rhs; }
    [ForceInline] This div(This rhs) { return this / rhs; }
    [ForceInline] This neg() { return -this; }
    
    [ForceInline] bool lt(This rhs) { return this < rhs; }
    [ForceInline] bool le(This rhs) { return this <= rhs; }
    [ForceInline] bool gt(This rhs) { return this > rhs; }
    [ForceInline] bool ge(This rhs) { return this >= rhs; }
    [ForceInline] bool eq(This rhs) { return this == rhs; }
    [ForceInline] bool ne(This rhs) { return this != rhs; }
    
    // Math functions - promote to float for precision, then convert back
    [ForceInline] This sin() { return half(::sin(float(this))); }
    [ForceInline] This cos() { return half(::cos(float(this))); }
    [ForceInline] This exp() { return half(::exp(float(this))); }
    [ForceInline] This log() { return half(::log(float(this))); }
    [ForceInline] This abs() { return half(::abs(float(this))); }
    [ForceInline] This sqrt() { return half(::sqrt(float(this))); }
    [ForceInline] This rsqrt() { return half(::rsqrt(float(this))); }
    [ForceInline] This floor() { return half(::floor(float(this))); }
    [ForceInline] This ceil() { return half(::ceil(float(this))); }
    [ForceInline] This tanh() { return half(::tanh(float(this))); }
    [ForceInline] This pow(This e) { return half(::pow(float(this), float(e))); }
    
    [ForceInline] static This max(This x, This y) { return half(::max(float(x), float(y))); }
    [ForceInline] static This min(This x, This y) { return half(::min(float(x), float(y))); }
    
    [ForceInline] This waveActiveSum() { return WaveActiveSum(this); }
}

extension half : ITensorElement
{
    typealias UnpackedType = half;
    
    [ForceInline] UnpackedType unpack() { return this; }
    [ForceInline] static This pack(UnpackedType value) { return value; }
}

// ============================================================================
// Extension: int (int32)
// ============================================================================

extension int : IUnpackedElement
{
    [ForceInline] __init(float x) { this = int(x); }
    
    [ForceInline] float toFloat() { return float(this); }
    [ForceInline] int toInt() { return this; }
    [ForceInline] static This fromFloat(float x) { return int(x); }
    [ForceInline] static This fromInt(int x) { return x; }
    
    [ForceInline] This add(This rhs) { return this + rhs; }
    [ForceInline] This sub(This rhs) { return this - rhs; }
    [ForceInline] This mul(This rhs) { return this * rhs; }
    [ForceInline] This div(This rhs) { return this / rhs; }
    [ForceInline] This neg() { return -this; }
    
    [ForceInline] bool lt(This rhs) { return this < rhs; }
    [ForceInline] bool le(This rhs) { return this <= rhs; }
    [ForceInline] bool gt(This rhs) { return this > rhs; }
    [ForceInline] bool ge(This rhs) { return this >= rhs; }
    [ForceInline] bool eq(This rhs) { return this == rhs; }
    [ForceInline] bool ne(This rhs) { return this != rhs; }
    
    // Math functions - for int, many of these just cast or are identity
    [ForceInline] This sin() { return int(::sin(float(this))); }
    [ForceInline] This cos() { return int(::cos(float(this))); }
    [ForceInline] This exp() { return int(::exp(float(this))); }
    [ForceInline] This log() { return int(::log(float(this))); }
    [ForceInline] This abs() { return ::abs(this); }
    [ForceInline] This sqrt() { return int(::sqrt(float(this))); }
    [ForceInline] This rsqrt() { return int(::rsqrt(float(this))); }
    [ForceInline] This floor() { return this; }
    [ForceInline] This ceil() { return this; }
    [ForceInline] This tanh() { return int(::tanh(float(this))); }
    [ForceInline] This pow(This e) { return int(::pow(float(this), float(e))); }
    
    [ForceInline] static This max(This x, This y) { return ::max(x, y); }
    [ForceInline] static This min(This x, This y) { return ::min(x, y); }
    
    [ForceInline] This waveActiveSum() { return WaveActiveSum(this); }
}

extension int : ITensorElement
{
    typealias UnpackedType = int;
    
    [ForceInline] UnpackedType unpack() { return this; }
    [ForceInline] static This pack(UnpackedType value) { return value; }
}

