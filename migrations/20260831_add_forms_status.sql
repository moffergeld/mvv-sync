-- Controls whether a player appears in the Wellness and RPE forms.
-- This is intentionally separate from is_active, which remains the GPS/reporting status.
alter table public.players
    add column if not exists forms_status text not null default 'actief';

alter table public.players
    drop constraint if exists players_forms_status_check;

alter table public.players
    add constraint players_forms_status_check
    check (forms_status in ('actief', 'inactief'));

-- Players already inactive for the squad must not be requested to complete forms.
update public.players
set forms_status = 'inactief'
where is_active is not true;

comment on column public.players.forms_status is
'Forms deelname: actief toont de speler in Wellness/RPE; inactief verbergt de speler uit Forms zonder GPS- of rapportagedata te wijzigen.';
