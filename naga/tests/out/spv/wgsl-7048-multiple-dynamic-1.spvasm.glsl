#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const vec3 _13[2] = vec3[](vec3(0.0), vec3(0.0));

void main()
{
    vec4 _19 = vec4(0.0);
    int _21 = 0;
    int _23 = 0;
    _19.x += (_13[_23].y * _13[_21].z);
}

