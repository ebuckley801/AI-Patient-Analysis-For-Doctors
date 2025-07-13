from dataclasses import dataclass
from typing import Optional

@dataclass
class Patient:
    patient_id: int
    patient_uid: str
    patient_note: str
    age: int
    gender: str
    id: Optional[int] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'patient_uid': self.patient_uid,
            'patient_note': self.patient_note,
            'age': self.age,
            'gender': self.gender
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data.get('id'),
            patient_id=data['patient_id'],
            patient_uid=data['patient_uid'],
            patient_note=data['patient_note'],
            age=data['age'],
            gender=data['gender']
        )