use glium::{implement_vertex, index::IndicesSource};
use nalgebra_glm as glm; 

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    xyz: [f32; 3], 
}

impl From<[f32; 3]> for Vertex {
    
    fn from(xyz: [f32; 3]) -> Self {
        Vertex {
            xyz
        }
    }

}

implement_vertex!(Vertex, xyz location(0)); 

pub const VERTEX_SHADER: &str = r#"
#version 330

layout (location = 0) in vec3 xyz; 

uniform mat4 view; 
uniform mat4 model; 
uniform mat4 projection; 
uniform mat4 world; 

void main() {
    gl_Position = projection * view * world * model * vec4(xyz, 1.0f); 
}
"#;

pub const FRAGMENT_SHADER: &str = r#"
#version 330

uniform vec4 rgb; 

out vec4 color; 

void main() {
    color = vec4(normalize(rgb.xyz), 0.75);
}
"#;

pub struct Plane {
    pub width: f32, 
    pub length: f32, 
    pub position: [f32; 3], 
    pub color: [f32; 4], 
    pub vertices: Vec<Vertex>, 
}

impl Plane {

    pub fn model(&self) -> [[f32; 4]; 4] {
        let [x, y, z] = self.position; 
        let model = glm::diagonal4x4(&glm::vec4(1.0f32, 1.0f32, 1.0f32, 1.0f32)); 
        let model = glm::translate(&model, &glm::vec3(x, y, z));
        let model = glm::scale(&model, &glm::vec3(self.width, 1.0f32, self.length)); 
        model.into()
    }

    pub fn indices(&self) -> impl Into<IndicesSource> {
        glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList)
    }

    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position = [x, y, z];
    }

}

impl Default for Plane {
    
    fn default() -> Self {
        Plane {
            width: 1.0, 
            length: 1.0, 
            position: [0.0, 0.0, 0.0], 
            color: [0.0, 0.0, 0.0, 0.2], 
            vertices: vec![
                [-0.5, 0.0, -0.5].into(), 
                [ 0.5, 0.0, -0.5].into(), 
                [ 0.5, 0.0,  0.5].into(), 
                [-0.5, 0.0, -0.5].into(), 
                [ 0.5, 0.0,  0.5].into(), 
                [-0.5, 0.0,  0.5].into()
            ]
        }
    }

}

