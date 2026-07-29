alter table public.players
    add column if not exists sub_position text;

comment on column public.players.sub_position is
'Vaste benchmark-subpositie voor dashboards en benchmarkvergelijkingen, bijvoorbeeld CB, LB, RB, DM, CM, AM, LW, RW, CF of GK.';
