#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint KX;
    uint KY;
    float param1;
    float param2;
} p;
