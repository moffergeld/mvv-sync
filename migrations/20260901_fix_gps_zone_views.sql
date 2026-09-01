-- Expose the canonical TD zone fields to every dashboard query.
-- The short zone aliases keep older report pages working during the transition.
create or replace view public.v_gps_summary as
select
  gps_id, player_id, player_name, datum, week, year, type, event, duration,
  total_distance_td as total_distance,
  td_zone_1_2 as walking, td_zone_3 as jogging, td_zone_4 as running, td_zone_5 as sprint, td_zone_6 as high_sprint,
  number_of_sprints, number_of_high_sprints, number_of_repeated_sprints, max_speed, avg_speed,
  playerload3d, playerload2d, total_accelerations, high_accelerations, total_decelerations, high_decelerations,
  hrzone1, hrzone2, hrzone3, hrzone4, hrzone5, hrtrimp, hrzoneanaerobic, avg_hr, max_hr, source_file, inserted_at,
  total_distance_td, td_zone_1, td_zone_2, td_zone_1_2, td_zone_3, td_zone_4, td_zone_5, td_zone_6,
  td_zone_1_2 as zone_1_2, td_zone_3 as zone_3, td_zone_4 as zone_4, td_zone_5 as zone_5, td_zone_6 as zone_6,
  heart_rate_exertion, csv_fatigue_index
from public.gps_records
where event = 'Summary';

create or replace view public.v_gps_match_events as
select
  gps_id, player_id, player_name, datum, week, year, type, event, duration,
  total_distance_td as total_distance,
  td_zone_1_2 as walking, td_zone_3 as jogging, td_zone_4 as running, td_zone_5 as sprint, td_zone_6 as high_sprint,
  number_of_sprints, number_of_high_sprints, number_of_repeated_sprints, max_speed, avg_speed,
  playerload3d, playerload2d, total_accelerations, high_accelerations, total_decelerations, high_decelerations,
  hrzone1, hrzone2, hrzone3, hrzone4, hrzone5, hrtrimp, hrzoneanaerobic, avg_hr, max_hr, source_file, inserted_at, match_id,
  total_distance_td, td_zone_1, td_zone_2, td_zone_1_2, td_zone_3, td_zone_4, td_zone_5, td_zone_6,
  td_zone_1_2 as zone_1_2, td_zone_3 as zone_3, td_zone_4 as zone_4, td_zone_5 as zone_5, td_zone_6 as zone_6,
  heart_rate_exertion, csv_fatigue_index
from public.gps_records gr
where match_id is not null
  and event <> 'Summary'
  and type = any (array['Match', 'Practice Match']);
