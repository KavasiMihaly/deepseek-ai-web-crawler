from pydantic import BaseModel


class EC(BaseModel):
    """
    Represents the data structure of a EnergyCertificate.
    """

    address: str
    energyrating: str
    potentialenergyrating: str
    totalfloorarea: int
    propertytype: str
    currentenergyscore: int
    potentialenergyscore: int
    currentco2emission: int
    potentialco2emission: int
    dateofcertificate: str
    energysaving: str