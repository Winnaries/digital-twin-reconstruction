use std::{io, collections::HashMap};
use csv::Error;
use glium::implement_vertex;
use serde::Deserialize;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Record {
    timestamp: u64, 
    device_id: String, 
    qw: f32, 
    qx: f32, 
    qy: f32, 
    qz: f32, 
    tx: f32, 
    ty: f32, 
    tz: f32, 
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sensor {
    rotation: [f32; 4], 
    translation: [f32; 3], 
}

implement_vertex!(Sensor, translation location(0), rotation location(1)); 

pub struct Trajectory {
    pub session_id: String,
    pub fragment_ids: Vec<String>,  
    pub sensors: Vec<Sensor>, 
}

pub fn parse_csv(reader: impl io::Read) -> io::Result<Vec<Trajectory>> {
    let reader = csv::ReaderBuilder::new().flexible(true).from_reader(reader); 
    let records: Vec<Record> = reader.into_deserialize().map(|x: Result<Record, Error>| x.unwrap()).collect();
    let mut table: HashMap<&str, Trajectory> = HashMap::new(); 

    records.iter().for_each(|r| {
        let device_id: Vec<&str> = r.device_id.split("/").collect(); 
        let (session_id, fragment_id) = (device_id[0], device_id[1]); 
        
        let sensor = Sensor {
            rotation: [r.qw, r.qx, r.qx, r.qz],
            translation: [r.tx, r.ty, r.tz],  
        }; 

        if let Some(traj) = table.get_mut(session_id) {
            traj.fragment_ids.push(fragment_id.to_owned()); 
            traj.sensors.push(sensor); 
        } else {
            table.insert(session_id, Trajectory { 
                session_id: session_id.to_owned(), 
                fragment_ids: vec![fragment_id.to_owned()], 
                sensors: vec![sensor],  
            }); 
        }
    });

    Ok(table.into_values().collect())
}

#[cfg(test)]
mod test {
    use std::ptr::read_unaligned;

    use super::*; 
    
    #[test]
    fn test_parse_csv() {
        let csv = "timestamp,device_id,qw,qx,qy,qz,tx,ty,tz
95378303839,ios_2021-06-02_14.31.38_000/cam_phone_95378303839,-0.2418468286681979,-0.9615713602902035,0.07694808168603257,-0.104735969263777,0.147598,0.057546,0.052738
95379336705,ios_2021-06-02_14.31.38_000/cam_phone_95379336705,-0.16862515492568173,-0.9620285988651034,0.04599729246165413,-0.20964441604329215,0.214302,0.069154,0.081037
95379936435,ios_2021-06-02_14.31.38_000/cam_phone_95379936435,-0.15116881409814376,-0.9125283171494514,0.04974920169651276,-0.3767825327362069,0.475122,0.079832,0.121503";

        let trajectories = parse_csv(csv.as_bytes()).unwrap();
        assert_eq!(trajectories.len(), 1); 
        
        let unaligned = std::ptr::addr_of!(trajectories[0].sensors[0].rotation[0]);
        unsafe { assert_eq!(read_unaligned(unaligned), -0.2418468286681979) }

        let unaligned = std::ptr::addr_of!(trajectories[0].sensors[2].translation[2]);
        unsafe { assert_eq!(read_unaligned(unaligned), 0.121503) }

    }

}