from typing import TypeAlias

Node: TypeAlias = int
Cluster: TypeAlias = set[Node]
Partition: TypeAlias = list[Cluster]