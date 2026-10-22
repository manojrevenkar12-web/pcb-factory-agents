-- The agent platform, the factory's own data, and two stand-ins for external systems.
-- Idempotent: safe to run on every start.

CREATE SCHEMA IF NOT EXISTS agents;   -- the platform every agent runs on
CREATE SCHEMA IF NOT EXISTS ops;      -- the factory's records (purchase orders, inbox, part library)
CREATE SCHEMA IF NOT EXISTS erp;      -- stand-in for the finance system (an external API)
CREATE SCHEMA IF NOT EXISTS mailprovider;  -- stand-in for the email provider (an external API)

-- Work items. A task is claimed under a lease; every claim bumps the fence, and only the
-- holder of the current fence may act or finish, so a worker that lost its lease cannot.
CREATE TABLE IF NOT EXISTS agents.tasks (
    id           bigserial PRIMARY KEY,
    kind         text        NOT NULL,
    business_key text        NOT NULL,
    payload      jsonb       NOT NULL DEFAULT '{}',
    status       text        NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'failed')),
    lease_owner  text,
    lease_until  timestamptz,
    fence        bigint      NOT NULL DEFAULT 0,
    attempts     int         NOT NULL DEFAULT 0,
    max_attempts int         NOT NULL DEFAULT 5,
    result       jsonb,
    error        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (kind, business_key)
);
CREATE INDEX IF NOT EXISTS tasks_claimable ON agents.tasks (kind, status, lease_until);

-- Durable receipts: one row per business action. A retried or resumed call with the same key
-- returns the stored result instead of acting again.
CREATE TABLE IF NOT EXISTS agents.receipts (
    key         text PRIMARY KEY,
    tool        text        NOT NULL,
    args_hash   text        NOT NULL,
    args        jsonb       NOT NULL,
    status      text        NOT NULL CHECK (status IN ('pending', 'done')),
    result      jsonb,
    task_id     bigint,
    session_id  text,
    attempts    int         NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

-- Every judge call, for cost and accuracy tracking per configuration.
CREATE TABLE IF NOT EXISTS agents.judge_calls (
    id            bigserial PRIMARY KEY,
    session_id    text,
    task_id       bigint,
    skill         text        NOT NULL,
    config_hash   text        NOT NULL,
    input_hash    text        NOT NULL,
    judge         text        NOT NULL,
    model         text        NOT NULL,
    input_tokens  int         NOT NULL DEFAULT 0,
    output_tokens int         NOT NULL DEFAULT 0,
    cost_usd      numeric(12, 6) NOT NULL DEFAULT 0,
    latency_ms    int         NOT NULL DEFAULT 0,
    cached        boolean     NOT NULL DEFAULT false,
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- A resumed session does not pay for the same judgement twice.
CREATE TABLE IF NOT EXISTS agents.judge_cache (
    config_hash text NOT NULL,
    input_hash  text NOT NULL,
    answer      jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (config_hash, input_hash)
);

-- Transactional outbox: an email is written in the same transaction as its receipt and
-- delivered later by the relay, which the provider de-duplicates by message id.
CREATE TABLE IF NOT EXISTS agents.outbox (
    message_id text PRIMARY KEY,
    recipient  text        NOT NULL,
    subject    text        NOT NULL,
    body       text        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    sent_at    timestamptz,
    attempts   int         NOT NULL DEFAULT 0
);

-- Cases an agent hands to a person, with the reason.
CREATE TABLE IF NOT EXISTS agents.reviews (
    id         bigserial PRIMARY KEY,
    key        text UNIQUE NOT NULL,
    kind       text        NOT NULL,
    reason     text        NOT NULL,
    payload    jsonb       NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    resolved   boolean     NOT NULL DEFAULT false
);

-- Shadow mode: what an agent would have done, recorded beside what the person did.
CREATE TABLE IF NOT EXISTS agents.shadow_actions (
    id          bigserial PRIMARY KEY,
    session_id  text        NOT NULL,
    tool        text        NOT NULL,
    key         text        NOT NULL,
    args        jsonb       NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tool, key)
);

-- ---- the factory's own records ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ops.suppliers (
    id    text PRIMARY KEY,
    name  text NOT NULL,
    email text NOT NULL
);

CREATE TABLE IF NOT EXISTS ops.purchase_orders (
    po_number     text PRIMARY KEY,
    supplier_id   text NOT NULL REFERENCES ops.suppliers (id),
    currency      text NOT NULL,
    ordered_on    date NOT NULL,
    promised_on   date NOT NULL
);

CREATE TABLE IF NOT EXISTS ops.po_lines (
    po_number  text           NOT NULL REFERENCES ops.purchase_orders (po_number),
    line_no    int            NOT NULL,
    mpn        text           NOT NULL,
    quantity   int            NOT NULL,
    unit_price numeric(12, 5) NOT NULL,
    PRIMARY KEY (po_number, line_no)
);

CREATE TABLE IF NOT EXISTS ops.goods_receipts (
    po_number   text NOT NULL REFERENCES ops.purchase_orders (po_number),
    mpn         text NOT NULL,
    quantity    int  NOT NULL,
    received_on date NOT NULL
);

CREATE TABLE IF NOT EXISTS ops.inbox (
    email_id    text PRIMARY KEY,
    sender      text        NOT NULL,
    subject     text        NOT NULL,
    body        text        NOT NULL,
    attachments jsonb       NOT NULL DEFAULT '[]',
    received_at timestamptz NOT NULL DEFAULT now()
);

-- What a person decided for past emails: the ground truth for replays and shadow runs.
CREATE TABLE IF NOT EXISTS ops.human_decisions (
    email_id text PRIMARY KEY REFERENCES ops.inbox (email_id),
    decision jsonb NOT NULL
);

-- The pick-and-place part library the part agent writes.
CREATE TABLE IF NOT EXISTS ops.pnp_parts (
    part_name  text PRIMARY KEY,
    record     jsonb       NOT NULL,
    written_at timestamptz NOT NULL DEFAULT now()
);

-- ---- external systems (stand-ins) --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS erp.idempotency (
    key        text PRIMARY KEY,
    body_hash  text  NOT NULL,
    response   jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS erp.bookings (
    id             bigserial PRIMARY KEY,
    idempotency_key text          NOT NULL,
    supplier_id    text           NOT NULL,
    invoice_number text           NOT NULL,
    po_number      text           NOT NULL,
    currency       text           NOT NULL,
    total          numeric(14, 2) NOT NULL,
    lines          jsonb          NOT NULL,
    booked_at      timestamptz    NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mailprovider.messages (
    message_id   text PRIMARY KEY,
    recipient    text        NOT NULL,
    subject      text        NOT NULL,
    body         text        NOT NULL,
    delivered_at timestamptz NOT NULL DEFAULT now()
);
