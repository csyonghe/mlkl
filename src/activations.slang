implementing mlkl;

interface IActivation
{
    static float apply(float x);
}

struct IdentityActivation : IActivation
{
    static float apply(float x) { return x; }
}

struct ReLUActivation : IActivation
{
    static float apply(float x) { return max(0.0, x); }
}

struct SiLUActivation : IActivation
{
    static float apply(float x) { return x * (1.0 / (1.0 + exp(-x))); }
}
