"""Message and role contract (Sprint 2, task 1).

Bon logical role dung chung backbone (INITIALIZATION.md §2.5); khac nhau boi
prompt, tool permission va role metadata. Module nay chi dinh nghia hop dong
trao doi message, khong trien khai analyst logic, Coordinator policy,
backbone, runner hay memory.

Ba loai message:

- ``AnalystRequest``: Coordinator → analyst (giao viec theo ngay).
- ``AnalystRecommendation``: analyst → Coordinator (de xuat, kem evidence).
- ``CoordinatorDecision``: Coordinator chot (duy nhat co quyen thuc thi).

Quy tac thuc thi:

- ``proposed_actions`` cua analyst chi la de xuat, KHONG duoc dua truc tiep
  vao ``ActionExecutor``. Chi ``final_actions`` cua ``CoordinatorDecision``
  moi duoc thuc thi. Dung ``executable_actions()`` de lay actions thuc thi
  duoc; ham nay raise ``TypeError`` cho moi message cua analyst.
- Business action tai dung union ``BusinessAction`` hien co
  (``mosaic.simulator.actions``); khong dinh nghia union thu hai.
- Tool evidence chi chap nhan 10 tool da trien khai trong
  ``mosaic.tools.api.ToolRunner``; memory tools Phase 3
  (``query_shared_graph``, ``write_private_memory``, ``propose_shared_fact``)
  chua trien khai nen bi tu choi o bien validate.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mosaic.simulator.actions import BusinessAction


class AgentRole(StrEnum):
    """Bon role khoa tai INITIALIZATION.md §2 va §20."""

    DEMAND = "demand"
    INVENTORY = "inventory"
    SUPPLIER_RISK = "supplier_risk"
    COORDINATOR = "coordinator"


ANALYST_ROLES: tuple[AgentRole, AgentRole, AgentRole] = (
    AgentRole.DEMAND,
    AgentRole.INVENTORY,
    AgentRole.SUPPLIER_RISK,
)

#: Muoi tool da trien khai trong ``ToolRunner._register_all`` (Sprint 1, task 10).
KNOWN_TOOLS: frozenset[str] = frozenset(
    {
        "get_inventory_state",
        "get_demand_history",
        "forecast_demand",
        "get_supplier_status",
        "get_open_orders",
        "get_shipments",
        "calculate_reorder_point",
        "calculate_transfer_cost",
        "calculate_order_cost",
        "simulate_action",
    }
)

#: Memory tools Phase 3 chua trien khai; liet ke de tu choi tuong minh.
UNIMPLEMENTED_MEMORY_TOOLS: frozenset[str] = frozenset(
    {
        "query_shared_graph",
        "write_private_memory",
        "propose_shared_fact",
    }
)

_ROLE_TOOL_MAP: dict[AgentRole, frozenset[str]] = {
    # Demand: read-only demand history and forecast; no simulation/action tool.
    AgentRole.DEMAND: frozenset({"get_demand_history", "forecast_demand"}),
    # Inventory: observed position/demand and read-only reorder/transfer quotes.
    AgentRole.INVENTORY: frozenset(
        {
            "get_inventory_state",
            "get_demand_history",
            "forecast_demand",
            "get_open_orders",
            "get_shipments",
            "calculate_reorder_point",
            "calculate_transfer_cost",
        }
    ),
    # Supplier-Risk reads supplier/order/shipment evidence and validates an
    # explicit inventory target; it cannot simulate outcomes or execute actions.
    AgentRole.SUPPLIER_RISK: frozenset(
        {
            "get_inventory_state",
            "get_supplier_status",
            "get_open_orders",
            "get_shipments",
            "calculate_order_cost",
        }
    ),
    # Coordinator: toan quyen 10 tool de tong hop va kiem chung what-if.
    AgentRole.COORDINATOR: frozenset(KNOWN_TOOLS),
}


def allowed_tools(role: AgentRole) -> frozenset[str]:
    """Tool duoc phep cho mot role (tap con cua ``KNOWN_TOOLS``)."""
    return _ROLE_TOOL_MAP[role]


def is_tool_allowed(role: AgentRole, tool: str) -> bool:
    """``True`` neu ``tool`` nam trong quyen cua ``role``."""
    allowed = _ROLE_TOOL_MAP.get(role)
    return allowed is not None and tool in allowed


def ensure_tool_allowed(role: AgentRole, tool: str) -> None:
    """Raise ``ValueError`` khi ``tool`` vuot quyen cua ``role``."""
    if tool in UNIMPLEMENTED_MEMORY_TOOLS:
        raise ValueError(f"Tool {tool!r} thuoc Phase 3, chua trien khai")
    if tool not in KNOWN_TOOLS:
        raise ValueError(f"Tool khong ton tai: {tool!r}")
    if not is_tool_allowed(role, tool):
        raise ValueError(f"Role {role.value!r} khong duoc dung tool {tool!r}")


def _empty_evidence() -> list[ToolEvidence]:
    return []


class ToolEvidence(BaseModel):
    """Mot tool call da chay, dung lam evidence cho recommendation/decision.

    Hinh dang khop ``ToolCallRecord`` cua single-agent runner de tai dung
    pattern hien co; ``tool`` phai thuoc ``KNOWN_TOOLS``.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    result: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_tool_known(self) -> ToolEvidence:
        if self.tool in UNIMPLEMENTED_MEMORY_TOOLS:
            raise ValueError(f"Tool {self.tool!r} thuoc Phase 3, chua trien khai")
        if self.tool not in KNOWN_TOOLS:
            raise ValueError(f"Tool khong ton tai: {self.tool!r}")
        return self


class AnalystRequest(BaseModel):
    """Yeu cau tu Coordinator toi mot analyst cho mot ngay mo phong."""

    model_config = ConfigDict(extra="forbid")

    message_type: Literal["request"] = "request"
    message_id: str = Field(min_length=1)
    episode_seed: int
    day: int = Field(ge=1)
    from_role: AgentRole = AgentRole.COORDINATOR
    to_role: AgentRole
    instruction: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_roles(self) -> AnalystRequest:
        if self.from_role is not AgentRole.COORDINATOR:
            raise ValueError("Request chi duoc phat hanh boi coordinator")
        if self.to_role not in ANALYST_ROLES:
            raise ValueError("Request phai gui toi mot analyst role")
        return self


class AnalystRecommendation(BaseModel):
    """De xuat cua analyst; ``proposed_actions`` chi la de xuat.

    Khong duoc dua ``proposed_actions`` truc tiep vao ``ActionExecutor``.
    """

    model_config = ConfigDict(extra="forbid")

    message_type: Literal["recommendation"] = "recommendation"
    message_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    episode_seed: int
    day: int = Field(ge=1)
    role: AgentRole
    summary: str = Field(min_length=1)
    proposed_actions: list[BusinessAction] = Field(min_length=1)
    evidence: list[ToolEvidence] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @model_validator(mode="after")
    def _validate_role_and_evidence(self) -> AnalystRecommendation:
        if self.role not in ANALYST_ROLES:
            raise ValueError("Recommendation chi duoc phat hanh boi analyst role")
        for item in self.evidence:
            ensure_tool_allowed(self.role, item.tool)
        return self


class CoordinatorDecision(BaseModel):
    """Quyet dinh cuoi cua Coordinator; duy nhat co quyen thuc thi."""

    model_config = ConfigDict(extra="forbid")

    message_type: Literal["decision"] = "decision"
    message_id: str = Field(min_length=1)
    episode_seed: int
    day: int = Field(ge=1)
    role: AgentRole = AgentRole.COORDINATOR
    recommendation_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    final_actions: list[BusinessAction] = Field(min_length=1)
    evidence: list[ToolEvidence] = Field(default_factory=_empty_evidence)

    @model_validator(mode="after")
    def _validate_role_and_evidence(self) -> CoordinatorDecision:
        if self.role is not AgentRole.COORDINATOR:
            raise ValueError("Final decision chi duoc phat hanh boi coordinator")
        for item in self.evidence:
            ensure_tool_allowed(AgentRole.COORDINATOR, item.tool)
        return self


type AnyMessage = AnalystRequest | AnalystRecommendation | CoordinatorDecision

MessageAdapter: TypeAdapter[AnyMessage] = TypeAdapter(
    Annotated[
        AnalystRequest | AnalystRecommendation | CoordinatorDecision,
        Field(discriminator="message_type"),
    ]
)


def executable_actions(message: AnyMessage) -> list[BusinessAction]:
    """Tra ve actions duoc phep thuc thi; analyst luon bi tu choi.

    Raise ``TypeError`` cho ``AnalystRequest``/``AnalystRecommendation`` de
    ngan analyst bi coi la executor cua business action.
    """
    if isinstance(message, CoordinatorDecision):
        return list(message.final_actions)
    if isinstance(message, AnalystRecommendation):
        role_name = message.role.value
    else:
        role_name = message.from_role.value
    raise TypeError(
        f"Message {message.message_type!r} cua role {role_name!r} khong co quyen "
        "thuc thi business action; chi CoordinatorDecision moi thuc thi duoc"
    )


def check_correlation(
    request: AnalystRequest,
    recommendation: AnalystRecommendation,
    decision: CoordinatorDecision,
) -> None:
    """Kiem tra tuong quan episode/day va lien ket id cua mot vong trao doi.

    Raise ``ValueError`` khi episode_seed/day lech nhau, ``request_id`` khong
    tro toi request, hoac decision khong tham chieu recommendation.
    """
    if not (request.episode_seed == recommendation.episode_seed == decision.episode_seed):
        raise ValueError("episode_seed khong khop giua request/recommendation/decision")
    if not (request.day == recommendation.day == decision.day):
        raise ValueError("day khong khop giua request/recommendation/decision")
    if recommendation.request_id != request.message_id:
        raise ValueError("recommendation.request_id khong tro toi request.message_id")
    if recommendation.message_id not in decision.recommendation_ids:
        raise ValueError("decision khong tham chieu recommendation.message_id")


def role_tool_permissions() -> dict[str, list[str]]:
    """Bang quyen tool theo role (da sap xep), de cong bo va kiem chung."""
    return {role.value: sorted(_ROLE_TOOL_MAP[role]) for role in AgentRole}


def export_schemas(output_dir: Path) -> list[Path]:
    """Xuat JSON schema da cong bo vao ``output_dir``; tra ve danh sach file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {
        "request.schema.json": AnalystRequest.model_json_schema(),
        "recommendation.schema.json": AnalystRecommendation.model_json_schema(),
        "decision.schema.json": CoordinatorDecision.model_json_schema(),
        "envelope.schema.json": MessageAdapter.json_schema(),
    }
    written: list[Path] = []
    for filename, schema in payloads.items():
        path = output_dir / filename
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    permissions_path = output_dir / "role-tools.json"
    permissions_path.write_text(
        json.dumps(role_tool_permissions(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(permissions_path)
    return written
