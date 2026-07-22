import json
from datetime import datetime, timedelta

events = [
  {"id":"jun2019", "start":"2019-06-01", "end":"2019-06-20"},
  {"id":"jul2019", "start":"2019-07-01", "end":"2019-07-20"},
  {"id":"jun2020", "start":"2020-06-01", "end":"2020-06-20"},
  {"id":"jul2020", "start":"2020-07-01", "end":"2020-07-20"},
  {"id":"may2021", "start":"2021-05-01", "end":"2021-05-15"},
  {"id":"jun2021", "start":"2021-06-01", "end":"2021-06-15"},
  {"id":"nov2021", "start":"2021-11-10", "end":"2021-11-20"},
  {"id":"may2022", "start":"2022-05-01", "end":"2022-05-15"},
  {"id":"jun2022", "start":"2022-06-01", "end":"2022-06-15"},
  {"id":"jul2022", "start":"2022-07-01", "end":"2022-07-15"},
  {"id":"jun2023", "start":"2023-06-01", "end":"2023-06-15"},
  {"id":"jul2023", "start":"2023-07-01", "end":"2023-07-15"},
  {"id":"sep2023", "start":"2023-09-15", "end":"2023-09-25"},
  {"id":"oct2023", "start":"2023-10-15", "end":"2023-10-25"},
  {"id":"feb2024", "start":"2024-02-10", "end":"2024-02-20"},
  {"id":"may2024", "start":"2024-05-15", "end":"2024-05-25"},
  {"id":"jan2025", "start":"2025-01-10", "end":"2025-01-20"},
  {"id":"feb2025", "start":"2025-02-05", "end":"2025-02-15"},
  {"id":"mar2025", "start":"2025-03-01", "end":"2025-03-10"},
  {"id":"sep2025_market", "start":"2025-09-12", "end":"2025-09-15"},
  {"id":"jun2025_lawoshime", "start":"2025-06-25", "end":"2025-07-05"},
  {"id":"may2026_downpour", "start":"2026-05-06", "end":"2026-05-18"},
  {"id":"jun2026_floodgates", "start":"2026-06-15", "end":"2026-07-03"}
]

events.sort(key=lambda x: x["start"])

for e in events:
    t_end = datetime.strptime(e["start"], "%Y-%m-%d")
    
    conflict = True
    while conflict:
        conflict = False
        t_start = t_end - timedelta(days=12)
        
        for prev in events:
            p_start = datetime.strptime(prev["start"], "%Y-%m-%d")
            if p_start >= datetime.strptime(e["start"], "%Y-%m-%d"):
                continue
                
            p_end_dirty = datetime.strptime(prev["end"], "%Y-%m-%d") + timedelta(days=14)
            
            if max(t_start, p_start) < min(t_end, p_end_dirty):
                conflict = True
                t_end = p_start
                break
                
    e["base_start"] = t_start.strftime("%Y-%m-%d")
    e["base_end"] = t_end.strftime("%Y-%m-%d")
    print(f"{e['id']:<20}: Event {e['start']}  -->  Baseline: {e['base_start']} to {e['base_end']}")
