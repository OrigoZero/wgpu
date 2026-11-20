#version 460

const vec2 _17[2] = vec2[](vec2(0.0), vec2(0.0));

layout(location = 0) out vec4 _11;

void main()
{
    int _21 = 0;
    _11 = (_17[_21] * _17[_21]).xxyy;
}

