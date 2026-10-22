"""What the invoice judge must answer. The schema is the contract between judgement and code."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Line(BaseModel):
    mpn: str = Field(description="Manufacturer part number exactly as written")
    quantity: int = Field(description="Units billed on this line")
    unit_price: float = Field(description="Price per unit in the invoice currency")


class InvoiceAnswer(BaseModel):
    document_type: Literal["invoice", "credit_note", "other"] = Field(
        description="invoice: a bill to pay. credit_note: money owed back. other: anything else, "
        "including order confirmations, shipping notices and quoted older invoices only"
    )
    invoice_number: str | None = Field(description="The supplier's invoice or credit note number")
    po_number: str | None = Field(description="Our purchase order number, digits only, e.g. 4500123")
    currency: Literal["EUR", "GBP", "USD"] | None
    lines: list[Line] = Field(description="Billed part lines; exclude shipping, freight and tax")
    shipping: float = Field(description="Shipping or freight charge, 0 if none")
    tax: float = Field(description="VAT or sales tax amount, 0 if none")
    total: float | None = Field(description="Total amount due including tax")
