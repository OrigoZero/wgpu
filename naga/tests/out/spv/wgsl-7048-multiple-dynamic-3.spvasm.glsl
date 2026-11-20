#version 460

struct _5
{
    float _m0;
    vec3 _m1;
};

const vec3 _93[12] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

_5 _12(vec3 _10[12], uint _11)
{
    vec3 _23 = vec3(0.0);
    vec3 _27 = vec3(0.0);
    uint _21 = 0u;
    uint _26 = 0u;
    uvec2 _41 = uvec2(4294967295u);
    uvec2 _67 = uvec2(4294967295u);
    vec3 _15[12] = _10;
    for (;;)
    {
        if (all(equal(uvec2(0u), _41)))
        {
            break;
        }
        _41 -= uvec2(uint(_41.y == 0u), 1u);
        if (!(_21 < _11))
        {
            break;
        }
        _23 = _15[_21];
        _21++;
        continue;
    }
    for (;;)
    {
        if (all(equal(uvec2(0u), _67)))
        {
            break;
        }
        _67 -= uvec2(uint(_67.y == 0u), 1u);
        if (!(_26 < _11))
        {
            break;
        }
        _27 = _15[_26];
        _26++;
        continue;
    }
    return _5(0.0, vec3(0.0));
}

void main()
{
}

