use glium::glutin::dpi::{LogicalSize, PhysicalPosition};
use glium::glutin::event::{Event, WindowEvent, VirtualKeyCode, ElementState, MouseButton};
use glium::glutin::{self, event_loop::ControlFlow};
use glium::{Surface, VertexBuffer, uniform};
use nalgebra_glm as glm;
use shader::{FRAGMENT_SHADER_SRC, VERTEX_SHADER_SRC};
use std::f32::consts::FRAC_PI_2;
use std::fs::File;
use rand::prelude::*; 

mod shader;
mod trajectory;
#[derive(Default)]

struct Movement {
    forward: bool, 
    backward: bool, 
    downward: bool, 
    upward: bool, 
    left: bool, 
    right: bool, 
}

struct Camera {
    yaw: f32, 
    pitch: f32,
    up: glm::Vec3, 
    position: glm::Vec3, 
}

impl Camera {
    fn look_at(&self) -> [[f32; 4]; 4] {
        let x = self.yaw.to_radians().cos() * self.pitch.to_radians().cos(); 
        let y = self.pitch.to_radians().sin(); 
        let z = self.yaw.to_radians().sin() * self.pitch.to_radians().cos(); 
        let front = glm::normalize(&glm::vec3(x, y, z)); 
        glm::look_at(&self.position, &(front + self.position), &self.up).into()
    }

    fn travel(&mut self, dx: f32, dz: f32) {
        let x = self.yaw.to_radians().cos() * self.pitch.to_radians().cos(); 
        let y = self.pitch.to_radians().sin(); 
        let z = self.yaw.to_radians().sin() * self.pitch.to_radians().cos(); 
        let front = glm::normalize(&glm::vec3(x, y, z)); 
        let right = glm::cross(&self.up, &front);
        self.position -= dx * right; 
        self.position -= dz * front; 
    }
}

impl Default for Camera {

    fn default() -> Self {
        Camera {
            yaw: -90.0, 
            pitch: 0.0, 
            up: glm::vec3(0.0, 1.0, 0.0), 
            position: glm::vec3(0.0, 0.0, 50.0), 
        }
    }

}

fn main() {
    let reader = File::open("./demo/trajectories.txt").expect("Failed to open the file.");
    let trajectories = trajectory::parse_csv(reader).expect("Failed to parse the CSV.");

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("TRAVIS")
        .with_inner_size(LogicalSize::new(1920 as u32, 1080 as u32));
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineStrip);
    let buffers: Vec<_> = trajectories
        .iter()
        .map(|x| &x.sensors)
        .map(|x| VertexBuffer::new(&display, x).unwrap())
        .collect();

    let mut holding = false; 
    let mut movement = Movement::default(); 
    let mut rng = rand::thread_rng(); 
    let mut prev_position: Option<PhysicalPosition<f64>> = None; 
    let colors: Vec<[f32; 3]> = buffers.iter().map(|_| [rng.gen(), rng.gen(), rng.gen()]).collect();

    let program =
        glium::Program::from_source(&display, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None)
            .unwrap();

    let speed = 0.1f32; 
    let mut current_time = std::time::Instant::now();
    let mut camera: Camera = Default::default(); 

    let model = glm::diagonal4x4(&glm::vec4(1.0f32, 1.0f32, 1.0f32, 1.0f32)); 
    let model = glm::rotate(&model, -FRAC_PI_2, &glm::vec3(1.0f32, 0.0f32, 0.0f32));
    let projection = glm::perspective(16.0 / 9.0, std::f32::consts::FRAC_PI_4, 0.1f32, 1000f32);

    event_loop.run(move |ev, _, control_flow| {
        let next_time = std::time::Instant::now(); 
        let elapsed = next_time.duration_since(current_time).as_millis(); 
        let next_frame_time = next_time + std::time::Duration::from_nanos(16_666_667);
        *control_flow = ControlFlow::WaitUntil(next_frame_time);
        current_time = next_time; 

        let distance = speed * elapsed as f32;
        if movement.forward   { camera.travel(0.0, -distance) }
        if movement.backward  { camera.travel(0.0, distance) }
        if movement.left      { camera.travel(-distance, 0.0) }
        if movement.right     { camera.travel(distance, 0.0) }
        if movement.downward  { camera.position.y -= distance }
        if movement.upward    { camera.position.y += distance }

        let projection: [[f32; 4]; 4] = projection.into(); 
        let model: [[f32; 4]; 4]= model.into(); 
        let view = camera.look_at(); 
        let mut target = display.draw(); 

        target.clear_color(0.2, 0.2, 0.3, 1.0);
        buffers.iter().zip(&colors).for_each(|(buf, &col)| {
            target
                .draw(
                    buf,
                    &indices,
                    &program,
                    &uniform! {
                        rgb: col, 
                        view: view, 
                        model: model, 
                        projection: projection, 
                    },
                    &Default::default(),
                )
                .unwrap()
        });

        target.finish().unwrap();

        match ev {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }, 
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(key) = input.virtual_keycode {
                        match key {
                            VirtualKeyCode::A => { movement.left = input.state == ElementState::Pressed },
                            VirtualKeyCode::D => { movement.right = input.state == ElementState::Pressed },
                            VirtualKeyCode::S => { movement.backward = input.state == ElementState::Pressed },
                            VirtualKeyCode::W => { movement.forward = input.state == ElementState::Pressed },
                            VirtualKeyCode::LShift => { movement.downward = input.state == ElementState::Pressed },
                            VirtualKeyCode::Space => { movement.upward = input.state == ElementState::Pressed },
                            _ => (), 
                        }
                    }
                },
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Right {
                        holding = state == ElementState::Pressed; 
                        if state == ElementState::Released {
                            prev_position = None; 
                        }
                    }
                },
                WindowEvent::CursorMoved { position, .. } => {
                    if holding {
                        if let Some(prev) = prev_position {
                            let (nx, ny) = (position.x, position.y); 
                            let (px, py) = (prev.x, prev.y);
                            let (dx, dy) = (nx - px, ny - py); 
                            
                            camera.yaw += 0.05 * dx as f32; 
                            camera.pitch -= 0.05 * dy as f32; 
                        } 
                    
                        prev_position = Some(position); 
                    }
                },
                _ => (),
            },
            _ => (),
        }
    });
}
