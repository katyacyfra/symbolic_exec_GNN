from dataclasses import dataclass

from dataclasses_json import dataclass_json
from typing import List

@dataclass_json
@dataclass
class StateHistoryElem:
    GraphVertexId: int
    NumOfVisits: int


@dataclass_json
@dataclass
class State:
    Id: int
    Position: int
    PredictedUsefulness: float
    PathConditionSize: int
    VisitedAgainVertices: int
    VisitedNotCoveredVerticesInZone: int
    VisitedNotCoveredVerticesOutOfZone: int
    History: List[StateHistoryElem]
    Children: List[int]

    def __hash__(self) -> int:
        return self.Id.__hash__()


@dataclass_json
@dataclass
class GameMapVertex:
    Uid: int
    Id: int
    InCoverageZone: bool
    BasicBlockSize: int
    CoveredByTest: bool
    VisitedByState: bool
    TouchedByState: bool
    States: List[int]


@dataclass_json
@dataclass
class GameEdgeLabel:
    Token: int


@dataclass_json
@dataclass
class GameMapEdge:
    VertexFrom: int
    VertexTo: int
    Label: GameEdgeLabel


@dataclass_json
@dataclass
class GameState:
    GraphVertices: List[GameMapVertex]
    States: List[State]
    Map: List[GameMapEdge]


@dataclass_json
@dataclass
class GameMap:
    Id: int
    MaxSteps: int
    CoverageToStart: int
    AssemblyFullName: str
    CoverageZone: bool
    NameOfObjectToCover: str

    def __hash__(self) -> int:
        return self.Id.__hash__()


@dataclass_json
@dataclass
class MoveReward:
    ForCoverage: int
    ForVisitedInstructions: int

    def __eq__(self, __o) -> bool:
        if type(self) != type(__o):
            raise RuntimeError(f"Can't compare {type(self)} with {type(__o)}")
        return self == __o

    def __add__(self, __o: "MoveReward") -> "MoveReward":
        if type(self) != type(__o):
            raise RuntimeError(f"Can't add {type(__o)} to {type(self)}")
        return MoveReward(
            self.ForCoverage + __o.ForCoverage,
            self.ForVisitedInstructions + __o.ForVisitedInstructions,
        )

    def printable(self, verbose=False) -> str:
        if verbose:
            return f"ForVisitedInstructions: {self.ForVisitedInstructions}"
        return f"#vi={self.ForVisitedInstructions}"


@dataclass_json
@dataclass
class Reward:
    ForMove: MoveReward
    MaxPossibleReward: int
