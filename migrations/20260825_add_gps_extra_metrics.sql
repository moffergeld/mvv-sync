alter table public.gps_records
    add column if not exists extra_metrics jsonb not null default '{}'::jsonb;

comment on column public.gps_records.extra_metrics is
'Additional vendor metrics from broad GPS/player exports that are not mapped to the canonical gps_records columns.';
