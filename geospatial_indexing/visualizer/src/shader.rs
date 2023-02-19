#![allow(dead_code)]

pub const VERTEX_SHADER_SRC: &str = r#"
#version 330

layout (location = 0) in vec3 position; 
layout (location = 1) in vec4 rotation; 

uniform mat4 view; 
uniform mat4 model; 
uniform mat4 projection; 
uniform mat4 world; 

vec4 quatmul(vec4 q1, vec4 q2) {
    vec4 qr;
    qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
    qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
    return qr;
}

void main() {
    gl_Position = projection * view * world * model * vec4(position, 1.0f); 
}
"#;

pub const FRAGMENT_SHADER_SRC: &str = r#"
#version 330

uniform vec3 rgb; 

out vec4 color; 

void main() {
    color = vec4(normalize(rgb), 0.8f);
}
"#;