// =====================================================================
// KETA COASTAL FLOOD PREDICTION -- PRODUCTION GEE SCRIPT v4.0
// Sentinel-1 SAR + FES2022 + ERA5 + CHIRPS + GPM IMERG + SMAP + SRTM
// Last updated: July 2026
//
// CHANGES FROM v3.2.15 (ARCHITECTURAL REDESIGN):
//
//   [CHANGE 1] Section 13b: CII (Coastal Inundation Index) redesigned.
//         Removed Dn = exp(-dist_water/1000) from CII. This term was
//         100% static terrain -- identical for every event -- and was
//         the primary reason CII was structurally high for every coastal
//         pixel regardless of actual tidal conditions. New CII = mean(Fn, Tn),
//         a purely tidal/elevation index. Fn (freeboard = tide_max - elev)
//         retains the physical tide-land interaction. The 0.5 threshold
//         is kept but operates on a 2-component mean instead of 3-component.
//
//   [CHANGE 2] Section 13b: Driver classes simplified from 5 to 3.
//         Old: 0=no-flood, 1=rain, 2=surge, 3=compound, 4=uncertain.
//         New: 0=no-flood, 1=rain-dominant, 2=coastal-dominant.
//         Rationale: (a) class 2 (surge) was never predicted on the test
//         set in v3.2.15 despite 856 actual pixels -- the model cannot
//         separate surge-only from compound on Keta's barrier; (b) the
//         5-class scheme required a 0.5 threshold on both indices,
//         creating a large "uncertain" bucket that absorbed genuinely-
//         driven pixels. New binary split: among flood==1 pixels, if
//         RFI >= CII -> rain-dominant (class 1), else coastal-dominant
//         (class 2). No threshold needed, no uncertain class, every
//         flooded pixel gets attributed.
//
//   [CHANGE 3] Section 26: Model B now trains on DYNAMIC-ONLY features.
//         Removed all 7 static terrain features (elevation, slope, aspect,
//         dist_water, lt3, lt1, lt0) from Model B's feature set. Model A
//         answers WHERE flooding occurs (terrain matters). Model B answers
//         WHY it flooded (weather/tides matter). These are architecturally
//         separate questions and should not share terrain features. The
//         old architecture had Model B predicting a terrain-derived label
//         from terrain features -- a tautology, not attribution.
//
//   [CHANGE 4] Section 22d: Event anomaly maps. Computes a mean baseline
//         probability map from all 6 built maps, then subtracts it from
//         each event's map. The anomaly map strips out the static terrain
//         pattern, revealing what is uniquely different about each event.
//         Addresses the "maps all look the same" problem -- raw maps
//         SHOULD look similar (terrain channels water identically), but
//         anomaly maps should differ between events with different
//         meteorological conditions.
//
//   [CHANGE 5] Section 28c: Strict leakage check. The v3.2.15 leakage
//         check kept 5 of 6 RFI/CII source variables (dist_water,
//         elevation, chirps_30d, soil_moisture, tide_range_fes) -- the
//         RF could reconstruct the label from what remained. New strict
//         check removes ALL variables that participate in RFI or CII
//         construction AND all terrain, testing with only truly
//         independent meteorological features (imerg_peak_intensity,
//         chirps_7d, solar_rad, wind components, season_wet,
//         spring_flag_fes). If accuracy holds, that IS genuine
//         independent attribution.
//
//   [NOTE] Model A is COMPLETELY UNCHANGED from v3.2.15. Same features,
//         same training, same RF, same evaluation. Only Model B and
//         diagnostics are modified.
//
//   [NOTE] All v3.2.15 diagnostics (6b S1 coverage, 21b terrain ablation,
//         22c map similarity, 24b/24c freeboard/imerg diagnostics) are
//         preserved unchanged.
// =====================================================================

// =====================================================================
// 1. ROI
// =====================================================================
var keta = ee.Geometry.Polygon([
  [[0.85, 5.95], [1.05, 5.95], [1.05, 5.85], [0.85, 5.85]]
]);
Map.centerObject(keta, 11);

// =====================================================================
// 2. STATIC FEATURES
// =====================================================================
var dem       = ee.Image("USGS/SRTMGL1_003");
var slope     = ee.Terrain.slope(dem);
var aspect    = ee.Terrain.aspect(dem);

var jrc       = ee.Image("JRC/GSW1_4/GlobalSurfaceWater");
var permWater = jrc.select("occurrence").gt(80);

var distWater = permWater
  .distance(ee.Kernel.euclidean(5000, "meters"))
  .reproject({crs: dem.projection(), scale: 90})
  .rename("dist_water");

var elev_lt3  = dem.lt(3).rename("lt3");
var elev_lt1  = dem.lt(1).rename("lt1");
var elev_lt0  = dem.lt(0).rename("lt0");

// =====================================================================
// 3. EVENT TABLE
// =====================================================================
var events = [
  {id:"jun2019", start:"2019-06-01", end:"2019-06-20", split:"train", confidence:"MEDIUM"},
  {id:"jul2019", start:"2019-07-01", end:"2019-07-20", split:"train", confidence:"MEDIUM"},
  {id:"jun2020", start:"2020-06-01", end:"2020-06-20", split:"train", confidence:"MEDIUM"},
  {id:"jul2020", start:"2020-07-01", end:"2020-07-20", split:"train", confidence:"MEDIUM"},
  {id:"may2021", start:"2021-05-15", end:"2021-06-01", split:"train", confidence:"HIGH"},
  {id:"jun2021", start:"2021-06-01", end:"2021-06-15", split:"train", confidence:"HIGH"},
  {id:"nov2021", start:"2021-11-05", end:"2021-11-15", split:"train", confidence:"HIGH"},
  {id:"may2022", start:"2022-05-15", end:"2022-06-01", split:"train", confidence:"MEDIUM"},
  {id:"jun2022", start:"2022-06-01", end:"2022-06-15", split:"train", confidence:"MEDIUM"},
  {id:"jul2022", start:"2022-07-01", end:"2022-07-31", split:"train", confidence:"MEDIUM"},
  {id:"jun2023", start:"2023-06-01", end:"2023-06-15", split:"test",  confidence:"HIGH"},
  {id:"jul2023", start:"2023-07-01", end:"2023-07-20", split:"test",  confidence:"HIGH"},
  {id:"sep2023", start:"2023-09-22", end:"2023-09-30", split:"test",  confidence:"HIGH"},
  {id:"oct2023", start:"2023-10-01", end:"2023-10-25", split:"test",  confidence:"HIGH"},
  {id:"feb2024", start:"2024-02-10", end:"2024-03-05", split:"validation", confidence:"HIGH"}, // [WIDENED 2026-07-13]
  {id:"may2024", start:"2024-05-01", end:"2024-05-20", split:"validation", confidence:"HIGH"},
  {id:"jan2025", start:"2025-01-16", end:"2025-01-28", split:"validation", confidence:"HIGH"},
  {id:"feb2025", start:"2025-02-01", end:"2025-02-08", split:"validation", confidence:"HIGH"},
  {id:"mar2025", start:"2025-03-01", end:"2025-03-25", split:"validation", confidence:"HIGH"},
  {id:"jun2025_lawoshime", start:"2025-06-25", end:"2025-07-05", split:"validation", confidence:"HIGH"},
  {id:"sep2025_market", start:"2025-09-12", end:"2025-09-15", split:"validation", confidence:"HIGH"},
  {id:"may2026_downpour", start:"2026-05-06", end:"2026-05-18", split:"validation", confidence:"HIGH"},
  {id:"jun2026_floodgates", start:"2026-06-15", end:"2026-07-03", split:"validation", confidence:"HIGH"}
];

var trainIds      = ["jun2019","jul2019","jun2020","jul2020","may2021","jun2021","nov2021","may2022","jun2022","jul2022"];
var testIds       = ["jun2023","jul2023","sep2023","oct2023"];
var validationIds = ["feb2024","may2024","jan2025","feb2025","mar2025","jun2025_lawoshime","sep2025_market","may2026_downpour","jun2026_floodgates"];

// =====================================================================
// 3b. BASELINE LOOKBACK CORRECTION (Generalized Rule)
// =====================================================================
events.sort(function(a, b) { return new Date(a.start) - new Date(b.start); });
events.forEach(function(e, i) {
  var tEnd = new Date(e.start);
  var conflict = true;
  while (conflict) {
    conflict = false;
    var tStart = new Date(tEnd.getTime() - 12 * 24 * 60 * 60 * 1000);
    for (var j = 0; j < i; j++) {
      var prev = events[j];
      var pStart = new Date(prev.start);
      if (pStart >= new Date(e.start)) continue;
      
      var pEndDirty = new Date(prev.end);
      pEndDirty.setDate(pEndDirty.getDate() + 14); // Clears the prior event by 14 days
      
      // Check for temporal intersection
      if (Math.max(tStart.getTime(), pStart.getTime()) < Math.min(tEnd.getTime(), pEndDirty.getTime())) {
        conflict = true;
        tEnd = new Date(pStart.getTime()); // Slide baseline to before the conflicting event
        break;
      }
    }
  }
  e.base_start = tStart.toISOString().split("T")[0];
  e.base_end   = tEnd.toISOString().split("T")[0];
});

// =====================================================================
// 4. TIDAL DICTIONARY (FES2022 -- sole tidal data source as of v3.2.3)
// =====================================================================
var FES2022_DATA = {
  "jun2019":{ max:0.712, range:1.528, spring:0 },
  "jul2019":{ max:0.734, range:1.588, spring:0 },
  "jun2020":{ max:0.792, range:1.618, spring:1 },
  "jul2020":{ max:0.647, range:1.429, spring:0 },
  "may2021":{ max:0.871, range:1.762, spring:1 },
  "jun2021":{ max:0.554, range:1.258, spring:0 },
  "nov2021":{ max:0.882, range:1.752, spring:1 },
  "may2022":{ max:0.838, range:1.711, spring:1 },
  "jun2022":{ max:0.770, range:1.659, spring:1 },
  "jul2022":{ max:0.726, range:1.608, spring:0 },
  "jun2023":{ max:0.709, range:1.551, spring:0 },
  "jul2023":{ max:0.695, range:1.572, spring:0 },
  "sep2023":{ max:0.884, range:1.673, spring:1 },
  "oct2023":{ max:0.880, range:1.634, spring:1 },
  "feb2024":{ max:0.884, range:1.793, spring:1 },
  "may2024":{ max:0.852, range:1.685, spring:1 },
  "jan2025":{ max:0.585, range:1.334, spring:0 },
  "feb2025":{ max:0.788, range:1.597, spring:0 },
  "mar2025":{ max:0.914, range:1.749, spring:1 },
  "jun2025_lawoshime":{ max:0.658, range:1.493, spring:0 },
  "sep2025_market":{ max:0.682, range:1.300, spring:0 },
  "may2026_downpour":{ max:0.821, range:1.694, spring:1 },
  "jun2026_floodgates":{ max:0.736, range:1.621, spring:1 }
};

// =====================================================================
// 5. SENTINEL-1 SAR COLLECTION (WITH ANGLE MASK)
// =====================================================================
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(keta)
  .filter(ee.Filter.eq("instrumentMode", "IW"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
  .map(function(img) {
    var angle = img.select("angle");
    return img.updateMask(angle.gt(30).and(angle.lt(45))).select("VH");
  });

// =====================================================================
// 6. FLOOD LABEL -- TRUE OTSU THRESHOLDING
// =====================================================================
function otsuThreshold(histogram) {
  var counts = ee.Array(ee.Dictionary(histogram).get('histogram'));
  var means  = ee.Array(ee.Dictionary(histogram).get('bucketMeans'));
  var size   = means.length().get([0]);
  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum    = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean   = sum.divide(total);

  var indices = ee.List.sequence(1, size);
  var bss = indices.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount  = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans  = means.slice(0, 0, i);
    var aMean   = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
    var bCount  = total.subtract(aCount);
    var safeBCount = ee.Number(ee.Algorithms.If(bCount.gt(0), bCount, 1));
    var bMean   = sum.subtract(aCount.multiply(aMean)).divide(safeBCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
      .add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });

  return means.sort(bss).get([-1]);
}

function getFloodLabel(event) {
  var t0 = ee.Date(event.get("start"));
  var t1 = ee.Date(event.get("end"));
  var bStart = ee.Date(event.get("base_start"));
  var bEnd   = ee.Date(event.get("base_end"));

  var beforeCol = s1.filterDate(bStart, bEnd);
  var afterCol  = s1.filterDate(t0, t1);

  var dummy = ee.Image.constant(-25).rename("VH").selfMask();

  var before = beforeCol.merge(ee.ImageCollection([dummy])).median().unmask(-25);
  var after  = afterCol.merge(ee.ImageCollection([dummy])).median().unmask(-25);

  before = before.focal_mean(30, "circle", "meters");
  after  = after.focal_mean(30,  "circle", "meters");

  var diff = after.subtract(before).rename("VH");

  var histResult = diff.reduceRegion({
    reducer:   ee.Reducer.histogram(255, 2),
    geometry:  keta,
    scale:     30,
    maxPixels: 1e9
  }).get("VH");

  var safeHist = ee.Dictionary(
    ee.Algorithms.If(
      histResult,
      histResult,
      ee.Dictionary({histogram: [1, 1], bucketMeans: [-2, -2]})
    )
  );

  var otsuThresh = ee.Number(
    ee.Algorithms.If(
      histResult,
      ee.Algorithms.If(
        ee.Array(ee.Dictionary(histResult).get('bucketMeans')).length().get([0]).gt(1),
        otsuThreshold(safeHist),
        -2
      ),
      -2
    )
  );

  var floodRaw = diff.lt(otsuThresh);
  var flood    = floodRaw
    .updateMask(permWater.not())
    .unmask(0)
    .rename("flood");

  return flood.set({
    "system:time_start": t0.millis(),
    "event_id": event.get("id"),
    "otsu_threshold": otsuThresh
  });
}

// =====================================================================
// 6b. SENTINEL-1 COVERAGE DIAGNOSTIC (run once, then comment out)
// =====================================================================
print("=== SENTINEL-1 COVERAGE PER EVENT (VH, angle-masked 30-45deg) ===");
print("before_images = count in the 12 days prior to event start");
print("after_images  = count in the event's start-to-end window");
events.forEach(function(e) {
  var t0 = ee.Date(e.start);
  var t1 = ee.Date(e.end);
  var beforeCount = s1.filterDate(t0.advance(-12, "day"), t0).size();
  var afterCount  = s1.filterDate(t0, t1).size();
  print(e.id + " (" + e.split + ", " + e.confidence + ")", ee.Dictionary({
    before_images: beforeCount,
    after_images: afterCount
  }));
});

// =====================================================================
// 7. GPM IMERG V7
// =====================================================================
var imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation");

// [FIX v3.2.12] getIMERG(t0) only ever looked at [t0-1day, t0] and
// [t0-3day, t0] -- i.e. the 1-3 days BEFORE the event's first day,
// never anything during the event window itself. Event windows in this
// script span up to 15-25 days (see section 3), so any heavy rain that
// actually caused a mid-event flood was structurally invisible to
// imerg_24h. 24c diagnostic (v3.2.11 run) confirmed this: 13/19 events
// had mean imerg_24h under 0.3mm, including confirmed flood events like
// jun2023 -- and jun2019, the one event whose mean crossed 25mm, did so
// purely because its 1-day pre-window happened to catch real rain by
// chance. Same "anchored to event start, never scans the event" bug
// family as the Sentinel-1 coverage fix (v3.2.5) and the freeboard fix
// (v3.2.7) already caught elsewhere in this pipeline.
//
// Fix: getIMERG now takes (t0, t1) -- event start AND end -- and scans
// the FULL window [t0-1day, t1]:
//   imerg_24h            = MAX single calendar-day (24h) rainfall total
//                           anywhere in the window (peak rain day of
//                           the event, via daily-binned sums + max --
//                           NOT a daily sum over whole event)
//   imerg_peak_intensity = MAX half-hourly rate anywhere in the window
//   imerg_3d              = TOTAL accumulated rainfall across the full
//                           event window (renamed in spirit only;
//                           band name kept as imerg_3d for compatibility)
// =====================================================================
function getIMERG(t0, t1) {
  var dummy = ee.Image.constant(0).rename("precipitation").selfMask();
  var windowStart = t0.advance(-1, "day");
  var nDays = t1.difference(windowStart, "day").ceil().max(1);
  var dayOffsets = ee.List.sequence(0, nDays.subtract(1));

  var dailySums = ee.ImageCollection.fromImages(
    dayOffsets.map(function(d) {
      var dayStart = windowStart.advance(d, "day");
      var dayEnd   = dayStart.advance(1, "day");
      var dayImgs  = imerg.filterDate(dayStart, dayEnd).select("precipitation")
        .merge(ee.ImageCollection([dummy]));
      return dayImgs.sum().multiply(0.5).unmask(0);
    })
  );

  var fullWindow = imerg.filterDate(windowStart, t1).select("precipitation")
    .merge(ee.ImageCollection([dummy]));

  var imerg24h  = dailySums.max().rename("imerg_24h");
  var imergPeak = fullWindow.max().unmask(0).rename("imerg_peak_intensity");
  var imerg3d   = dailySums.sum().rename("imerg_3d");
  return ee.Image.cat([imerg24h, imergPeak, imerg3d]);
}

// =====================================================================
// 8. CHIRPS
// =====================================================================
var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY");

function getRain(t) {
  var dummy = ee.Image.constant(0).rename("precipitation").selfMask();

  var rain7d = chirps.filterDate(t.advance(-7,  "day"), t).select("precipitation")
    .merge(ee.ImageCollection([dummy])).sum().unmask(0).rename("chirps_7d");
  var rain30d = chirps.filterDate(t.advance(-30, "day"), t).select("precipitation")
    .merge(ee.ImageCollection([dummy])).sum().unmask(0).rename("chirps_30d");
  return ee.Image.cat([rain7d, rain30d]);
}

// =====================================================================
// 9. ERA5 (v3.2.5 -- switched to ERA5/HOURLY, aggregated to daily)
// =====================================================================
var era5 = ee.ImageCollection("ECMWF/ERA5/HOURLY");

function getERA5(t) {
  var c = era5.filterDate(t.advance(-3, "day"), t);

  var dummyU     = ee.Image.constant(0).rename("u_component_of_wind_10m").selfMask();
  var dummyV     = ee.Image.constant(0).rename("v_component_of_wind_10m").selfMask();
  var dummySolar = ee.Image.constant(0).rename("surface_net_solar_radiation").selfMask();

  var u = c.select("u_component_of_wind_10m").merge(ee.ImageCollection([dummyU])).mean().unmask(0).rename("u_wind_10m");
  var v = c.select("v_component_of_wind_10m").merge(ee.ImageCollection([dummyV])).mean().unmask(0).rename("v_wind_10m");
  var solar = c.select("surface_net_solar_radiation").merge(ee.ImageCollection([dummySolar]))
    .mean().divide(3600).unmask(0).rename("solar_rad");

  var windSpeed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed");
  var windDir   = u.atan2(v).multiply(180 / Math.PI).rename("wind_dir");
  return ee.Image.cat([solar, u, v, windSpeed, windDir]);
}

// =====================================================================
// 10. SMAP SOIL MOISTURE
// =====================================================================
var smap = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture");

function getSM(t) {
  var col = smap.filterDate(t.advance(-5, "day"), t).select("ssm");
  var dummy = ee.Image.constant(0.2).rename("ssm").selfMask();
  return col.merge(ee.ImageCollection([dummy])).mean().unmask(0.2).rename("soil_moisture");
}

// =====================================================================
// 11. HYDRODYNAMIC BANDS
// =====================================================================
function getHydroBands(eventId) {
  var fes = ee.Dictionary(ee.Dictionary(FES2022_DATA).get(eventId));
  return ee.Image.cat([
    ee.Image.constant(ee.Number(fes.get("max"))).rename("tide_max_fes"),
    ee.Image.constant(ee.Number(fes.get("range"))).rename("tide_range_fes"),
    ee.Image.constant(ee.Number(fes.get("spring"))).rename("spring_flag_fes")
  ]);
}

// =====================================================================
// 11b. FREEBOARD BAND (v3.2.7)
// =====================================================================
function getFreeboard(hydro, elevationImg) {
  return hydro.select("tide_max_fes").subtract(elevationImg).rename("freeboard_fes");
}

// =====================================================================
// 12. COMPOUND RISK BAND (v3.2.7 -- continuous score, not hard AND)
// =====================================================================
function getCompoundRisk(imergBands, sm, hydro) {
  var normRain = imergBands.select("imerg_24h").divide(50).clamp(0, 1);
  var normSM   = sm.subtract(0.1).divide(0.4).clamp(0, 1);
  var normTide = hydro.select("tide_range_fes").subtract(1.258).divide(1.793 - 1.258).clamp(0, 1);
  return normRain.multiply(normSM).multiply(normTide).rename("compound_risk").toFloat();
}

// =====================================================================
// 13. SEASON FLAG (Fixed Integer Casting)
// =====================================================================
function getSeasonFlag(t) {
  var month = ee.Number(t.get("month")).toInt();
  var wet   = month.gte(4).and(month.lte(6)).or(month.gte(9).and(month.lte(11)));
  return ee.Image.constant(ee.Number(wet).toInt()).rename("season_wet").toFloat();
}

// =====================================================================
// 13a. PERCENTILE STATS FOR RFI/CII (2026-07-14)
//   The continuous attribution indices below need P10/P90 percentiles
//   to normalize each raw variable onto 0-1. Computed here from a
//   lightweight pooled sample across all 19 events, using the raw
//   source functions (getIMERG, getRain, getSM, getHydroBands,
//   getFreeboard) DIRECTLY rather than going through build()/
//   getDriverLabel -- build() is what constructs the samples that
//   would otherwise be the obvious source, but build() itself calls
//   getDriverLabel, which needs these percentiles already in hand.
//   Doing it this way avoids that circular dependency. ~150 random
//   points/event is enough for stable P10/P90 without risking the
//   server timeout the heavier 24c per-event loop hit once already.
// =====================================================================
var rawIndexSamples = ee.FeatureCollection(events.map(function(e) {
  var t0 = ee.Date(e.start);
  var t1 = ee.Date(e.end);
  var imergBands = getIMERG(t0, t1);
  var rain       = getRain(t0);
  var sm         = getSM(t0);
  var hydro      = getHydroBands(e.id);
  var freeboard  = getFreeboard(hydro, dem);

  var rawImg = ee.Image.cat([
    imergBands.select("imerg_24h"),
    rain.select("chirps_30d"),
    sm,
    freeboard,
    hydro.select("tide_range_fes")
  ]).clip(keta);

  return rawImg.sample({region: keta, scale: 200, numPixels: 150, seed: 42, geometries: false});
})).flatten();

function getPct1090(bandName) {
  return rawIndexSamples.reduceColumns({
    reducer: ee.Reducer.percentile([10, 90]),
    selectors: [bandName]
  });
}

var pctImerg  = getPct1090("imerg_24h");
var pctChirps = getPct1090("chirps_30d");
var pctSm     = getPct1090("soil_moisture");
var pctFb     = getPct1090("freeboard_fes");
var pctTr     = getPct1090("tide_range_fes");

var rfciiPct = {
  imerg_p10:  ee.Number(pctImerg.get("p10")),  imerg_p90:  ee.Number(pctImerg.get("p90")),
  chirps_p10: ee.Number(pctChirps.get("p10")), chirps_p90: ee.Number(pctChirps.get("p90")),
  sm_p10:     ee.Number(pctSm.get("p10")),     sm_p90:     ee.Number(pctSm.get("p90")),
  fb_p10:     ee.Number(pctFb.get("p10")),     fb_p90:     ee.Number(pctFb.get("p90")),
  tr_p10:     ee.Number(pctTr.get("p10")),     tr_p90:     ee.Number(pctTr.get("p90"))
};

print("=== 13a: RFI/CII normalization percentiles (P10/P90) ===");
print(ee.String("imerg_24h:      p10=").cat(rfciiPct.imerg_p10.format("%.3f")).cat("  p90=").cat(rfciiPct.imerg_p90.format("%.3f")));
print(ee.String("chirps_30d:     p10=").cat(rfciiPct.chirps_p10.format("%.3f")).cat("  p90=").cat(rfciiPct.chirps_p90.format("%.3f")));
print(ee.String("soil_moisture:  p10=").cat(rfciiPct.sm_p10.format("%.3f")).cat("  p90=").cat(rfciiPct.sm_p90.format("%.3f")));
print(ee.String("freeboard_fes:  p10=").cat(rfciiPct.fb_p10.format("%.3f")).cat("  p90=").cat(rfciiPct.fb_p90.format("%.3f")));
print(ee.String("tide_range_fes: p10=").cat(rfciiPct.tr_p10.format("%.3f")).cat("  p90=").cat(rfciiPct.tr_p90.format("%.3f")));

// =====================================================================
// 13b. DRIVER ATTRIBUTION LABEL (Model B -- REVISED v4.0)
//   RFI (Rainfall Flooding Index) = mean(Ri, Ra, S)   -- UNCHANGED
//     Ri = normalized imerg_24h
//     Ra = normalized chirps_30d
//     S  = normalized soil_moisture
//   CII (Coastal Inundation Index) = mean(Fn, Tn)      -- CHANGED
//     Fn = normalized freeboard_fes (tide_max - elevation)
//     Tn = normalized tide_range_fes
//     [REMOVED] Dn = exp(-dist_water/1000) was 100% static terrain.
//
//   Class assignment (SIMPLIFIED from 5 to 3):
//     0 = no flood        (flood==0)
//     1 = rain-dominant   (flood==1, RFI >= CII)
//     2 = coastal-dominant(flood==1, CII > RFI)
//   Every flooded pixel gets attributed. No uncertain class.
//   RFI and CII are kept as bands for downstream analysis.
// =====================================================================
function getDriverLabelV2(floodBand, imergBands, chirpsBand, smBand, freeboardBand, tideRangeBand, pct) {
  var isFlood = floodBand.eq(1);

  function norm01(img, p10, p90) {
    return img.subtract(p10).divide(p90.subtract(p10)).clamp(0, 1);
  }

  var Ri = norm01(imergBands.select("imerg_24h"), pct.imerg_p10, pct.imerg_p90);
  var Ra = norm01(chirpsBand.select("chirps_30d"), pct.chirps_p10, pct.chirps_p90);
  var S  = norm01(smBand, pct.sm_p10, pct.sm_p90);
  var RFI = Ri.add(Ra).add(S).divide(3).rename("RFI");

  var Fn = norm01(freeboardBand, pct.fb_p10, pct.fb_p90);
  var Tn = norm01(tideRangeBand, pct.tr_p10, pct.tr_p90);
  var CII = Fn.add(Tn).divide(2).rename("CII");

  // Simple binary attribution: among flooded pixels, higher index wins
  var rainDominant   = isFlood.and(RFI.gte(CII));  // class 1
  var coastDominant  = isFlood.and(CII.gt(RFI));    // class 2

  var driver = ee.Image(0)
    .where(rainDominant, 1)
    .where(coastDominant, 2)
    .rename("flood_driver")
    .toInt();

  return ee.Image.cat([driver, RFI.toFloat(), CII.toFloat()]);
}

// =====================================================================
// 14. FEATURE STACK BUILDER
// =====================================================================
function build(event) {
  var t0         = ee.Date(event.get("start"));
  var t1         = ee.Date(event.get("end")); // [ADD v3.2.12] needed by getIMERG's widened window
  var id         = event.get("id");
  var imergBands = getIMERG(t0, t1);
  var rain       = getRain(t0);
  var era        = getERA5(t0);
  var sm         = getSM(t0);
  var hydro      = getHydroBands(id);
  var label      = getFloodLabel(event);
  var season     = getSeasonFlag(t0);
  var compound   = getCompoundRisk(imergBands, sm, hydro);
  var freeboard  = getFreeboard(hydro, dem);
  var driverLabel = getDriverLabelV2(label, imergBands, rain, sm, freeboard, hydro.select("tide_range_fes"), rfciiPct);

  var terrain = ee.Image.cat([
    dem.rename("elevation"), slope.rename("slope"), aspect.rename("aspect"),
    distWater, elev_lt3, elev_lt1, elev_lt0
  ]);

  return ee.Image.cat([
    terrain,
    imergBands,
    rain,
    era,
    sm,
    hydro,
    freeboard,
    season,
    compound
  ]).addBands(label).addBands(driverLabel).clip(keta).set('event_id', id);
}

// =====================================================================
// DIAGNOSTIC (optional -- run once, then comment out).
// =====================================================================
var checkEvent = ee.Date("2023-06-01");
var eraCheck = era5.filterDate(checkEvent.advance(-3, "day"), checkEvent);
print("ERA5/HOURLY image count for jun2023 window (expect ~72, 24hr x 3d):", eraCheck.size());

print("ERA5/HOURLY pixel count with valid wind data in ROI (should be > 0):",
  eraCheck.select("u_component_of_wind_10m").mean().reduceRegion({
    reducer: ee.Reducer.count(), geometry: keta, scale: 27830,
    maxPixels: 1e9, bestEffort: true
  })
);

print("ERA5/HOURLY wind min/max over ROI (should NOT be null/0-only):",
  eraCheck.select(["u_component_of_wind_10m", "v_component_of_wind_10m"]).mean().reduceRegion({
    reducer: ee.Reducer.minMax(), geometry: keta, scale: 27830,
    maxPixels: 1e9, bestEffort: true
  })
);

print("[diag] Band names on a single ERA5/HOURLY image:", eraCheck.first().bandNames());

print("[diag] temperature_2m pixel count in ROI (control -- known-good band):",
  eraCheck.select("temperature_2m").mean().reduceRegion({
    reducer: ee.Reducer.count(), geometry: keta, scale: 27830,
    maxPixels: 1e9, bestEffort: true
  })
);

print("[diag] u_component_of_wind_10m at ROI CENTROID (point, native 27830m scale):",
  eraCheck.select("u_component_of_wind_10m").mean().reduceRegion({
    reducer: ee.Reducer.first(), geometry: keta.centroid(1),
    scale: 27830, bestEffort: true
  })
);

print("[diag] u_component_of_wind_10m pixel count over a 50km-buffered ROI:",
  eraCheck.select("u_component_of_wind_10m").mean().reduceRegion({
    reducer: ee.Reducer.count(), geometry: keta.buffer(50000), scale: 27830,
    maxPixels: 1e9, bestEffort: true
  })
);

// =====================================================================
// 15. SAMPLE GENERATION (Scale=200 + Crash Fallback)
// =====================================================================
var nPtsMap = ee.Dictionary({HIGH: 500, MEDIUM: 200});

var samples = ee.FeatureCollection(
  ee.List(events).map(function(e) {
    e = ee.Dictionary(e);
    var img  = build(e);
    var nPts = ee.Number(nPtsMap.get(e.get("confidence")));

    var floodSum = img.select("flood").reduceRegion({
      reducer: ee.Reducer.sum(), geometry: keta, scale: 200, maxPixels: 1e9
    }).getNumber("flood");

    var samp = ee.Algorithms.If(
      floodSum.gt(0),
      img.stratifiedSample({
        numPoints: nPts, classBand: "flood", region: keta,
        scale: 200, geometries: true, seed: 42
      }),
      img.sample({
        numPixels: nPts, region: keta,
        scale: 200, geometries: true, seed: 42
      })
    );

    return ee.FeatureCollection(samp).map(function(f) {
      return f.set("event_id", e.get("id")).set("split", e.get("split")).set("confidence", e.get("confidence"));
    });
  })
).flatten();

// =====================================================================
// 16. TRAIN / TEST / VALIDATION SPLITS
// =====================================================================
var train      = samples.filter(ee.Filter.inList("event_id", trainIds));
var test       = samples.filter(ee.Filter.inList("event_id", testIds));
var validation = samples.filter(ee.Filter.inList("event_id", validationIds));

// =====================================================================
// 16b. dist_water DIAGNOSTIC (2026-07-13)
// =====================================================================
print("=== dist_water BY CLASS (all 19 events, meters) ===");
print("flood == 1 (should reveal whether any inland flood samples exist):",
  samples.filter(ee.Filter.eq("flood", 1))
    .reduceColumns({
      reducer: ee.Reducer.percentile([0, 10, 25, 50, 75, 90, 100]),
      selectors: ["dist_water"]
    })
);
print("flood == 0 (for comparison):",
  samples.filter(ee.Filter.eq("flood", 0))
    .reduceColumns({
      reducer: ee.Reducer.percentile([0, 10, 25, 50, 75, 90, 100]),
      selectors: ["dist_water"]
    })
);

// =====================================================================
// 17. CLASS BALANCING (v3.2.3 -- dynamic minority oversampling)
// =====================================================================
var floodSamples    = train.filter(ee.Filter.eq("flood", 1));
var nonFloodSamples = train.filter(ee.Filter.eq("flood", 0));

var floodCount    = floodSamples.size();
var nonFloodCount = nonFloodSamples.size();

print("=== TRAINING SET CLASS COUNTS (pre-balance) ===");
print("Flood samples:    ", floodCount);
print("Non-flood samples:", nonFloodCount);

var minorIsFlood = floodCount.lt(nonFloodCount);

var minorCollection = ee.FeatureCollection(
  ee.Algorithms.If(minorIsFlood, floodSamples, nonFloodSamples)
);
var majorCollection = ee.FeatureCollection(
  ee.Algorithms.If(minorIsFlood, nonFloodSamples, floodSamples)
);

var minorCount = ee.Number(ee.Algorithms.If(minorIsFlood, floodCount, nonFloodCount));
var majorCount = ee.Number(ee.Algorithms.If(minorIsFlood, nonFloodCount, floodCount));

var safeMinorCount = ee.Number(ee.Algorithms.If(minorCount.gt(0), minorCount, 1));
var repeatFactor = majorCount.divide(safeMinorCount).ceil().max(1).toInt();

var minorRepeated = ee.FeatureCollection(
  ee.List.sequence(1, repeatFactor).map(function(i) {
    return minorCollection;
  })
).flatten();

var minorBalanced = minorRepeated
  .randomColumn("rand_balance", 42)
  .sort("rand_balance")
  .limit(majorCount);

var balancedTrain = majorCollection.merge(minorBalanced);

print("=== TRAINING SET (post-balance) ===");
print("Minority class was: ", ee.Algorithms.If(minorIsFlood, "flood", "non-flood"));
print("Majority count:     ", majorCount);
print("Minority count (bal):", minorBalanced.size());
print("Balanced total:      ", balancedTrain.size());

// =====================================================================
// 18. FEATURE SETS (v3.2.7 -- added freeboard_fes)
// =====================================================================
var inputFeatures = [
  "elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0",
  "imerg_24h", "imerg_peak_intensity", "imerg_3d",
  "chirps_7d", "chirps_30d",
  "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
  "soil_moisture",
  "tide_max_fes", "tide_range_fes", "spring_flag_fes", "freeboard_fes",
  "compound_risk", "season_wet"
];

// =====================================================================
// 18b. MODEL B FEATURE SET -- DYNAMIC ONLY (v4.0)
//   Model B answers "WHY did this pixel flood?" -- that's a weather/
//   tide question, not a terrain question. Removing all 7 static terrain
//   features forces the RF to learn from meteorological conditions
//   rather than memorizing which geographic locations historically
//   clustered into each driver class.
// =====================================================================
var inputFeaturesDriverOnly = [
  "imerg_24h", "imerg_peak_intensity", "imerg_3d",
  "chirps_7d", "chirps_30d",
  "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
  "soil_moisture",
  "tide_max_fes", "tide_range_fes", "spring_flag_fes", "freeboard_fes",
  "compound_risk", "season_wet"
];

// =====================================================================
// 19. RF MODEL (Probability Output)
// =====================================================================
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).setOutputMode('PROBABILITY')
.train({
  features: balancedTrain,
  classProperty: "flood",
  inputProperties: inputFeatures
});

print("=== FEATURE IMPORTANCE ===");
print(rf.explain().get("importance"));

// =====================================================================
// 19b. FEATURE IMPORTANCE DIAGNOSTIC (sorted table for readability)
// =====================================================================
var importanceDict = ee.Dictionary(rf.explain().get("importance"));
var impKeys = importanceDict.keys();
var impVals = impKeys.map(function(k) { return importanceDict.get(k); });

var impFC = ee.FeatureCollection(
  impKeys.zip(impVals).map(function(pair) {
    pair = ee.List(pair);
    return ee.Feature(null, {
      band: pair.get(0),
      importance: pair.get(1)
    });
  })
).sort("importance", false);

print("=== FEATURE IMPORTANCE (sorted, highest first) ===");
print(impFC);

var impStrings = impFC.toList(impFC.size()).map(function(f) {
  f = ee.Feature(f);
  return ee.String(f.get('band')).cat(': ').cat(ee.Number(f.get('importance')).format('%.4f'));
});
print("=== FEATURE IMPORTANCE (band: score, easy-copy) ===", impStrings);

// =====================================================================
// 20. TEST SET EVALUATION (Probability Thresholding)
// =====================================================================
var testClassified = test.classify(rf).map(function(f) {
  var prob = ee.Number(f.get("classification"));
  return f.set("flood_pred", prob.gte(0.5).toInt());
});

var testMatrix = testClassified.errorMatrix("flood", "flood_pred");

print("=== TEST SET (2023) ===");
print("Confusion Matrix:", testMatrix);
print("Overall Accuracy:", testMatrix.accuracy());
print("Kappa:", testMatrix.kappa());

var testRecall    = testMatrix.producersAccuracy().get([1, 0]);
var testPrecision = testMatrix.consumersAccuracy().get([0, 1]);
var testF1 = ee.Number(2).multiply(testPrecision.multiply(testRecall))
                        .divide(testPrecision.add(testRecall));
print("Sensitivity (Recall):", testRecall);
print("Precision:", testPrecision);
print("F1-Score:", testF1);

// =====================================================================
// 21. VALIDATION SET EVALUATION (2024-2025)
// =====================================================================
var valClassified = validation.classify(rf).map(function(f) {
  var prob = ee.Number(f.get("classification"));
  return f.set("flood_pred", prob.gte(0.5).toInt());
});
var valMatrix = valClassified.errorMatrix("flood", "flood_pred");
print("=== VALIDATION SET (2024-2025) ===");
print("Confusion Matrix:", valMatrix);
print("Overall Accuracy:", valMatrix.accuracy());

var recall = valMatrix.producersAccuracy().get([1, 0]);
var precision = valMatrix.consumersAccuracy().get([0, 1]);
var f1 = ee.Number(2).multiply(precision.multiply(recall))
                    .divide(precision.add(recall));

print("Sensitivity (Recall):", recall);
print("Precision:", precision);
print("F1-Score:", f1);

// =====================================================================
// 21b. MODEL A TERRAIN-ABLATION CHECK (2026-07-14)
//   Feature importance ranked slope (1037), aspect (919), and dist_water
//   (899) as the top 3 of 23 features -- all static terrain, identical
//   for every event regardless of that event's actual rain/tide/wind
//   conditions. If Model A leans this heavily on features that never
//   vary between events, its probability maps across different events
//   would structurally look similar (same low-lying-land-near-water
//   footprint every time), which is exactly what prompted this check --
//   the validation maps visually looked alike across very different
//   events. Retrains Model A WITHOUT slope/aspect/dist_water and
//   re-evaluates the same way (sections 20-21) for direct comparison.
//   elevation and lt3/lt1/lt0 are left in -- they're also static terrain,
//   but weren't the flagged top-3, so removing them is out of scope for
//   this specific check.
//   Small accuracy drop: model still discriminates well from
//   event-varying features alone -- terrain was informative but not the
//   whole story. Large drop: Model A is substantially a static
//   susceptibility map with event conditions doing little of the work.
// =====================================================================
var inputFeaturesNoTerrain = inputFeatures.filter(function(f) {
  return f !== "slope" && f !== "aspect" && f !== "dist_water";
});

var rfNoTerrain = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).setOutputMode('PROBABILITY')
.train({
  features: balancedTrain,
  classProperty: "flood",
  inputProperties: inputFeaturesNoTerrain
});

var testNoTerrainClassified = test.classify(rfNoTerrain).map(function(f) {
  var prob = ee.Number(f.get("classification"));
  return f.set("flood_pred", prob.gte(0.5).toInt());
});
var testNoTerrainMatrix = testNoTerrainClassified.errorMatrix("flood", "flood_pred");

print("=== 21b: MODEL A TERRAIN-ABLATED, TEST SET (2023) ===");
print("Overall Accuracy:", testNoTerrainMatrix.accuracy());
print("Kappa:", testNoTerrainMatrix.kappa());

var valNoTerrainClassified = validation.classify(rfNoTerrain).map(function(f) {
  var prob = ee.Number(f.get("classification"));
  return f.set("flood_pred", prob.gte(0.5).toInt());
});
var valNoTerrainMatrix = valNoTerrainClassified.errorMatrix("flood", "flood_pred");

print("=== 21b: MODEL A TERRAIN-ABLATED, VALIDATION SET (2024-2025) ===");
print("Overall Accuracy:", valNoTerrainMatrix.accuracy());
print("Kappa:", valNoTerrainMatrix.kappa());

print("=== 21b: HOW TO READ THIS ===");
print("Compare against the full-feature Model A above (test ~0.93,");
print("validation ~0.94). A drop of only a few points means rain/tide/");
print("wind/soil alone still carry most of the discriminative power --");
print("terrain adds refinement, it isn't the whole model. A large drop");
print("means Model A leans heavily on static terrain, and event-specific");
print("conditions are contributing comparatively little -- consistent");
print("with maps that look visually similar across very different events.");

// =====================================================================
// 22. MAP VISUALISATION
// =====================================================================
function buildProbMap(eventDict) {
  var img = build(eventDict);
  return img.select(inputFeatures).classify(rf);
}

var vizEvent_jun2023 = ee.Dictionary({
  id:"jun2023", start:"2023-06-01", end:"2023-06-15",
  split:"test", confidence:"HIGH"
});
var vizImg_jun2023  = build(vizEvent_jun2023);
var vizProb_jun2023 = buildProbMap(vizEvent_jun2023);

var vizEvent_feb2024 = ee.Dictionary({
  id:"feb2024", start:"2024-02-10", end:"2024-03-05",
  split:"validation", confidence:"HIGH"
});
var vizProb_feb2024 = buildProbMap(vizEvent_feb2024);

var vizEvent_may2024 = ee.Dictionary({
  id:"may2024", start:"2024-05-01", end:"2024-05-20",
  split:"validation", confidence:"HIGH"
});
var vizProb_may2024 = buildProbMap(vizEvent_may2024);

var vizEvent_jan2025 = ee.Dictionary({
  id:"jan2025", start:"2025-01-16", end:"2025-01-28",
  split:"validation", confidence:"HIGH"
});
var vizProb_jan2025 = buildProbMap(vizEvent_jan2025);

var vizEvent_feb2025 = ee.Dictionary({
  id:"feb2025", start:"2025-02-01", end:"2025-02-08",
  split:"validation", confidence:"HIGH"
});
var vizProb_feb2025 = buildProbMap(vizEvent_feb2025);

var vizEvent_mar2025 = ee.Dictionary({
  id:"mar2025", start:"2025-03-01", end:"2025-03-25",
  split:"validation", confidence:"HIGH"
});
var vizProb_mar2025 = buildProbMap(vizEvent_mar2025);

Map.addLayer(vizImg_jun2023.select("elevation"),
  {min:0, max:10, palette:["blue","green","yellow","red"]}, "Elevation (m)", false);
Map.addLayer(vizImg_jun2023.select("freeboard_fes"),
  {min:-5, max:2, palette:["000080","4169E1","ADD8E6","FFFFFF","FFA500","FF0000"]},
  "Freeboard (tide_max - elevation, m)", false);
Map.addLayer(vizImg_jun2023.select("imerg_24h"),
  {min:0, max:100, palette:["white","cyan","blue"]}, "IMERG 24h Rain (mm)", false);
Map.addLayer(permWater.selfMask(),
  {palette:["0000FF"]}, "Permanent Water (JRC)", false);

// =====================================================================
// 22b. jun2023 GROUND-TRUTH + THRESHOLD DIAGNOSTIC (2026-07-13)
// =====================================================================
Map.addLayer(vizImg_jun2023.select("flood").selfMask(),
  {palette: ["FF00FF"]}, "RAW SAR flood label (ground truth, jun2023)", true);

var masked_jun2023_lowThresh = vizProb_jun2023
  .updateMask(permWater.not())
  .updateMask(vizProb_jun2023.gt(0.02));
Map.addLayer(masked_jun2023_lowThresh,
  {min:0.02, max:1, palette:["FFFFCC", "FFA500", "FF0000"]},
  "Flood Prob: Jun 2023 (LOW threshold 0.02)", false);

var probVis = {min:0.1, max:1, palette:["FFFF00", "FFA500", "FF0000"]};

var masked_jun2023 = vizProb_jun2023.updateMask(permWater.not()).updateMask(vizProb_jun2023.gt(0.1));
var masked_feb2024 = vizProb_feb2024.updateMask(permWater.not()).updateMask(vizProb_feb2024.gt(0.1));
var masked_may2024 = vizProb_may2024.updateMask(permWater.not()).updateMask(vizProb_may2024.gt(0.1));
var masked_jan2025 = vizProb_jan2025.updateMask(permWater.not()).updateMask(vizProb_jan2025.gt(0.1));
var masked_feb2025 = vizProb_feb2025.updateMask(permWater.not()).updateMask(vizProb_feb2025.gt(0.1));
var masked_mar2025 = vizProb_mar2025.updateMask(permWater.not()).updateMask(vizProb_mar2025.gt(0.1));

Map.addLayer(masked_jun2023, probVis, "Flood Prob: Jun 2023 (Test)", true);
Map.addLayer(masked_feb2024, probVis, "Flood Prob: Feb 2024", false);
Map.addLayer(masked_may2024, probVis, "Flood Prob: May 2024", false);
Map.addLayer(masked_jan2025, probVis, "Flood Prob: Jan 2025", false);
Map.addLayer(masked_feb2025, probVis, "Flood Prob: Feb 2025", false);
Map.addLayer(masked_mar2025, probVis, "Flood Prob: Mar 2025", false);

// =====================================================================
// 22c. MAP SIMILARITY DIAGNOSTIC (2026-07-14)
//   Companion to 21b -- that checks accuracy impact of removing terrain;
//   this checks the actual visual-similarity complaint directly. For
//   every pair among the 6 probability maps already built above,
//   computes mean absolute pixel-wise difference and Pearson correlation
//   over the whole ROI. High correlation + low mean abs diff between
//   maps from events with genuinely different rain/tide conditions would
//   confirm the maps really are structurally near-identical, not just
//   visually similar at a glance. Uses "classification" as the band
//   name -- that's classify()'s default output band name, matching
//   sections 20/21's testClassified/valClassified pattern.
// =====================================================================
var probMaps = {
  jun2023: vizProb_jun2023, feb2024: vizProb_feb2024, may2024: vizProb_may2024,
  jan2025: vizProb_jan2025, feb2025: vizProb_feb2025, mar2025: vizProb_mar2025
};
var probKeys = Object.keys(probMaps);

print("=== 22c: PAIRWISE PROBABILITY MAP SIMILARITY (all 6 built maps) ===");
print("meanAbsDiff near 0 + correlation near 1 = maps are structurally");
print("near-identical despite different event conditions.");

for (var pi = 0; pi < probKeys.length; pi++) {
  for (var pj = pi + 1; pj < probKeys.length; pj++) {
    var keyA = probKeys[pi];
    var keyB = probKeys[pj];
    var imgA = probMaps[keyA];
    var imgB = probMaps[keyB];

    var meanAbsDiff = imgA.subtract(imgB).abs().reduceRegion({
      reducer: ee.Reducer.mean(), geometry: keta, scale: 200, maxPixels: 1e9
    }).getNumber("classification");

    var corrDict = ee.Image.cat([imgA.rename("a"), imgB.rename("b")]).reduceRegion({
      reducer: ee.Reducer.pearsonsCorrelation(), geometry: keta, scale: 200, maxPixels: 1e9
    });
    var corr = ee.Number(corrDict.get("correlation"));

    print(ee.String(keyA).cat(" vs ").cat(keyB).cat(":  meanAbsDiff=")
      .cat(meanAbsDiff.format("%.4f")).cat("  correlation=").cat(corr.format("%.4f")));
  }
}

// =====================================================================
// 22d. EVENT ANOMALY MAPS (v4.0)
//   Raw probability maps look similar because terrain dominates the
//   spatial pattern. Anomaly maps subtract the mean baseline to reveal
//   what is UNIQUE about each event -- areas that are unusually high-risk
//   or low-risk compared to the historical average.
// =====================================================================
var meanProbMap = vizProb_jun2023
  .add(vizProb_feb2024)
  .add(vizProb_may2024)
  .add(vizProb_jan2025)
  .add(vizProb_feb2025)
  .add(vizProb_mar2025)
  .divide(6);

var anomaly_jun2023 = vizProb_jun2023.subtract(meanProbMap);
var anomaly_feb2024 = vizProb_feb2024.subtract(meanProbMap);
var anomaly_may2024 = vizProb_may2024.subtract(meanProbMap);
var anomaly_jan2025 = vizProb_jan2025.subtract(meanProbMap);
var anomaly_feb2025 = vizProb_feb2025.subtract(meanProbMap);
var anomaly_mar2025 = vizProb_mar2025.subtract(meanProbMap);

var anomalyVis = {min: -0.3, max: 0.3, palette: ["0000FF", "FFFFFF", "FF0000"]};
// Blue = lower risk than average, White = average, Red = higher risk than average

Map.addLayer(anomaly_jun2023, anomalyVis, "ANOMALY: Jun 2023 (vs mean)", false);
Map.addLayer(anomaly_feb2024, anomalyVis, "ANOMALY: Feb 2024 (vs mean)", false);
Map.addLayer(anomaly_may2024, anomalyVis, "ANOMALY: May 2024 (vs mean)", false);
Map.addLayer(anomaly_jan2025, anomalyVis, "ANOMALY: Jan 2025 (vs mean)", false);
Map.addLayer(anomaly_feb2025, anomalyVis, "ANOMALY: Feb 2025 (vs mean)", false);
Map.addLayer(anomaly_mar2025, anomalyVis, "ANOMALY: Mar 2025 (vs mean)", false);

Map.addLayer(meanProbMap.updateMask(permWater.not()).updateMask(meanProbMap.gt(0.1)),
  probVis, "BASELINE: Mean Probability Map (all 6 events)", false);

print("=== 22d: ANOMALY MAP STATISTICS ===");
print("Positive anomaly = higher flood risk than average for that pixel.");
print("Negative anomaly = lower flood risk than average.");
print("If anomaly maps differ visually across events, the model IS");
print("responding to event-specific conditions despite similar raw maps.");

var anomalyMaps = {
  jun2023: anomaly_jun2023, feb2024: anomaly_feb2024, may2024: anomaly_may2024,
  jan2025: anomaly_jan2025, feb2025: anomaly_feb2025, mar2025: anomaly_mar2025
};
var anomalyKeys = Object.keys(anomalyMaps);

for (var ai = 0; ai < anomalyKeys.length; ai++) {
  var aKey = anomalyKeys[ai];
  var aImg = anomalyMaps[aKey];
  var aStats = aImg.reduceRegion({
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true}),
    geometry: keta, scale: 200, maxPixels: 1e9
  });
  print(ee.String(aKey).cat(":  mean=").cat(ee.Number(aStats.get("classification_mean")).format("%.4f"))
    .cat("  stdDev=").cat(ee.Number(aStats.get("classification_stdDev")).format("%.4f"))
    .cat("  min=").cat(ee.Number(aStats.get("classification_min")).format("%.4f"))
    .cat("  max=").cat(ee.Number(aStats.get("classification_max")).format("%.4f")));
}

// Pairwise correlation of ANOMALY maps (should be LOWER than raw maps
// if the model is truly responding to event-specific conditions)
print("=== 22d: PAIRWISE ANOMALY MAP CORRELATION ===");
print("Compare against 22c raw-map correlations (all >0.97).");
print("Lower anomaly correlations = events produce genuinely different risk patterns.");
for (var ani = 0; ani < anomalyKeys.length; ani++) {
  for (var anj = ani + 1; anj < anomalyKeys.length; anj++) {
    var anKeyA = anomalyKeys[ani];
    var anKeyB = anomalyKeys[anj];
    var anCorrDict = ee.Image.cat([
      anomalyMaps[anKeyA].rename("a"), anomalyMaps[anKeyB].rename("b")
    ]).reduceRegion({
      reducer: ee.Reducer.pearsonsCorrelation(), geometry: keta, scale: 200, maxPixels: 1e9
    });
    var anCorr = ee.Number(anCorrDict.get("correlation"));
    print(ee.String(anKeyA).cat(" vs ").cat(anKeyB).cat(":  anomaly_correlation=")
      .cat(anCorr.format("%.4f")));
  }
}

// =====================================================================
// 23. EXPORTS (Model A)
// =====================================================================
Export.table.toDrive({
  collection: samples,
  description: "keta_samples_v4_0_all_splits",
  fileFormat: "CSV"
});

Export.image.toDrive({
  image: vizProb_jun2023.toFloat(),
  description: "keta_flood_probability_jun2023_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_feb2024.toFloat(),
  description: "keta_flood_probability_feb2024_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_may2024.toFloat(),
  description: "keta_flood_probability_may2024_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_jan2025.toFloat(),
  description: "keta_flood_probability_jan2025_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_feb2025.toFloat(),
  description: "keta_flood_probability_feb2025_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_mar2025.toFloat(),
  description: "keta_flood_probability_mar2025_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

// =====================================================================
// =====================================================================
// MODEL B: COMPOUND FLOOD DRIVER ATTRIBUTION (v4.0)
// =====================================================================

// =====================================================================
// 24. DRIVER-STRATIFIED SAMPLE GENERATION
// =====================================================================
var samplesDriver = ee.FeatureCollection(
  ee.List(events).map(function(e) {
    e = ee.Dictionary(e);
    var img  = build(e);
    var nPts = ee.Number(nPtsMap.get(e.get("confidence")));

    var driverSum = img.select("flood_driver").reduceRegion({
      reducer: ee.Reducer.sum(), geometry: keta, scale: 200, maxPixels: 1e9
    }).getNumber("flood_driver");

    var samp = ee.Algorithms.If(
      driverSum.gt(0),
      img.stratifiedSample({
        numPoints: nPts, classBand: "flood_driver", region: keta,
        scale: 200, geometries: true, seed: 42
      }),
      img.sample({
        numPixels: nPts, region: keta,
        scale: 200, geometries: true, seed: 42
      })
    );

    return ee.FeatureCollection(samp).map(function(f) {
      return f.set("event_id", e.get("id")).set("split", e.get("split")).set("confidence", e.get("confidence"));
    });
  })
).flatten();

var trainDriver      = samplesDriver.filter(ee.Filter.inList("event_id", trainIds));
var testDriver       = samplesDriver.filter(ee.Filter.inList("event_id", testIds));
var validationDriver = samplesDriver.filter(ee.Filter.inList("event_id", validationIds));

print("=== MODEL B: flood_driver CLASS COUNTS (pre-balance, training set) ===");
print("0 = no-flood:          ", trainDriver.filter(ee.Filter.eq("flood_driver", 0)).size());
print("1 = rain-dominant:     ", trainDriver.filter(ee.Filter.eq("flood_driver", 1)).size());
print("2 = coastal-dominant:  ", trainDriver.filter(ee.Filter.eq("flood_driver", 2)).size());

// =====================================================================
// 24c. IMERG_24H SPATIAL-VS-EVENT-CONSTANT DIAGNOSTIC (2026-07-13)
// =====================================================================
var allEventIds = trainIds.concat(testIds).concat(validationIds);

var imergByEventLines = ee.List(allEventIds.map(function(id) {
  var evtSamples = samplesDriver.filter(ee.Filter.eq("event_id", id));
  var stats = evtSamples.reduceColumns({
    reducer: ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true}),
    selectors: ["imerg_24h"]
  });
  return ee.String(id).cat("  n=").cat(ee.Number(evtSamples.size()).format("%d"))
    .cat("  mean=").cat(ee.Number(stats.get("mean")).format("%.2f"))
    .cat("  stdDev=").cat(ee.Number(stats.get("stdDev")).format("%.2f"))
    .cat("  min=").cat(ee.Number(stats.get("min")).format("%.2f"))
    .cat("  max=").cat(ee.Number(stats.get("max")).format("%.2f"));
}));

print("=== 24c: imerg_24h mean/stdDev/min/max per event, all 19 events ===");
print("[v3.2.12] This now reads from the FIXED getIMERG() -- window scans");
print("the full event span, not just 1 day before event start.");
print(ee.String(imergByEventLines.join("\n")));

// =====================================================================
// 24b. FREEBOARD / DEM SANITY DIAGNOSTIC (2026-07-13)
// =====================================================================
var fbFlood0 = samplesDriver.filter(ee.Filter.eq("flood", 0))
  .reduceColumns({
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
    selectors: ["freeboard_fes"]
  });
var fbFlood1 = samplesDriver.filter(ee.Filter.eq("flood", 1))
  .reduceColumns({
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
    selectors: ["freeboard_fes"]
  });
print("=== 24b-Q1: Freeboard, flood==0 -- mean ===", fbFlood0.get("mean"));
print("=== 24b-Q1: Freeboard, flood==0 -- stdDev ===", fbFlood0.get("stdDev"));
print("=== 24b-Q1: Freeboard, flood==1 -- mean ===", fbFlood1.get("mean"));
print("=== 24b-Q1: Freeboard, flood==1 -- stdDev ===", fbFlood1.get("stdDev"));

var elevStatsAll = dem.clip(keta).reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95]).combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }),
  geometry: keta, scale: 30, maxPixels: 1e9
});
print("=== 24b-Q2a whole ROI: p5 ===", elevStatsAll.get("elevation_p5"));
print("=== 24b-Q2a whole ROI: p25 ===", elevStatsAll.get("elevation_p25"));
print("=== 24b-Q2a whole ROI: p50 ===", elevStatsAll.get("elevation_p50"));
print("=== 24b-Q2a whole ROI: p75 ===", elevStatsAll.get("elevation_p75"));
print("=== 24b-Q2a whole ROI: p95 ===", elevStatsAll.get("elevation_p95"));
print("=== 24b-Q2a whole ROI: stdDev ===", elevStatsAll.get("elevation_stdDev"));

var nearWaterMask = distWater.lte(100);
var farWaterMask  = distWater.gte(500);

var elevNearWater = dem.updateMask(nearWaterMask).clip(keta).reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95]).combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }),
  geometry: keta, scale: 30, maxPixels: 1e9
});
print("=== 24b-Q2b <=100m from water: p5 ===", elevNearWater.get("elevation_p5"));
print("=== 24b-Q2b <=100m from water: p25 ===", elevNearWater.get("elevation_p25"));
print("=== 24b-Q2b <=100m from water: p50 ===", elevNearWater.get("elevation_p50"));
print("=== 24b-Q2b <=100m from water: p75 ===", elevNearWater.get("elevation_p75"));
print("=== 24b-Q2b <=100m from water: p95 ===", elevNearWater.get("elevation_p95"));
print("=== 24b-Q2b <=100m from water: stdDev ===", elevNearWater.get("elevation_stdDev"));

var elevFarWater = dem.updateMask(farWaterMask).clip(keta).reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95]).combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }),
  geometry: keta, scale: 30, maxPixels: 1e9
});
print("=== 24b-Q2b >=500m from water: p5 ===", elevFarWater.get("elevation_p5"));
print("=== 24b-Q2b >=500m from water: p25 ===", elevFarWater.get("elevation_p25"));
print("=== 24b-Q2b >=500m from water: p50 ===", elevFarWater.get("elevation_p50"));
print("=== 24b-Q2b >=500m from water: p75 ===", elevFarWater.get("elevation_p75"));
print("=== 24b-Q2b >=500m from water: p95 ===", elevFarWater.get("elevation_p95"));
print("=== 24b-Q2b >=500m from water: stdDev ===", elevFarWater.get("elevation_stdDev"));

print(ui.Chart.image.histogram({
  image: dem.clip(keta), region: keta, scale: 30, maxBuckets: 40
}).setOptions({title: "24b-Q2c: Elevation histogram, whole ROI"}));

// =====================================================================
// 25. N-CLASS BALANCING (generalized from Model A's 2-class version)
// =====================================================================
var driverClasses = [0, 1, 2]; // [CHANGE v4.0] simplified to 3 classes

var driverCollectionsByClass = driverClasses.map(function(c) {
  return trainDriver.filter(ee.Filter.eq("flood_driver", c));
});
var driverCountsByClass = driverCollectionsByClass.map(function(fc) {
  return fc.size();
});

var maxDriverCount = ee.Number(driverCountsByClass[0])
  .max(driverCountsByClass[1]).max(driverCountsByClass[2]);

var balancedDriverParts = driverClasses.map(function(c, i) {
  var coll  = driverCollectionsByClass[i];
  var count = ee.Number(driverCountsByClass[i]);
  var safeCount = ee.Number(ee.Algorithms.If(count.gt(0), count, 1));
  var repeatFactor = maxDriverCount.divide(safeCount).ceil().max(1).toInt();

  var repeated = ee.FeatureCollection(
    ee.List.sequence(1, repeatFactor).map(function(i) { return coll; })
  ).flatten();

  return ee.Algorithms.If(
    count.gt(0),
    repeated.randomColumn("rand_balance_driver", 42).sort("rand_balance_driver").limit(maxDriverCount),
    ee.FeatureCollection([]) // class absent from training data -- empty, not null
  );
});

var balancedTrainDriver = driverClasses.reduce(function(acc, c, i) {
  var part = ee.FeatureCollection(balancedDriverParts[i]);
  return acc.merge(part);
}, ee.FeatureCollection([]));

print("=== MODEL B: TRAINING SET (post-balance) ===");
print("Balanced total:", balancedTrainDriver.size());
print("NOTE: if any class printed 0 above (section 24), it was entirely");
print("absent from your 10 training events -- check whether the driver-");
print("labeling thresholds (RFI, CII, see section 13a/13b) are");
print("realistic for this dataset before trusting Model B's results.");

// =====================================================================
// 26. MODEL B RF (multiclass, default classification mode)
// =====================================================================
var rfDriver = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).train({
  features: balancedTrainDriver,
  classProperty: "flood_driver",
  inputProperties: inputFeaturesDriverOnly
});

print("=== MODEL B: FEATURE IMPORTANCE ===");
print(rfDriver.explain().get("importance"));

// =====================================================================
// 27. MODEL B TEST SET EVALUATION (3x3 confusion matrix)
// =====================================================================
var testDriverClassified = testDriver.classify(rfDriver, "driver_pred");
var testDriverMatrix = testDriverClassified.errorMatrix("flood_driver", "driver_pred");

print("=== MODEL B TEST SET (2023) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2):", testDriverMatrix);
print("Overall Accuracy:", testDriverMatrix.accuracy());
print("Kappa:", testDriverMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", testDriverMatrix.producersAccuracy());
print("Per-class Consumer's Accuracy (precision):", testDriverMatrix.consumersAccuracy());

// =====================================================================
// 28. MODEL B VALIDATION SET EVALUATION (2024-2025)
// =====================================================================
var valDriverClassified = validationDriver.classify(rfDriver, "driver_pred");
var valDriverMatrix = valDriverClassified.errorMatrix("flood_driver", "driver_pred");

print("=== MODEL B VALIDATION SET (2024-2025) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2):", valDriverMatrix);
print("Overall Accuracy:", valDriverMatrix.accuracy());
print("Kappa:", valDriverMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", valDriverMatrix.producersAccuracy());
print("Per-class Consumer's Accuracy (precision):", valDriverMatrix.consumersAccuracy());

// =====================================================================
// 28b. MODEL B LEAKAGE CHECK (2026-07-14)
//   This retrains Model B WITHOUT imerg_24h, freeboard_fes, tide_max_fes
//   (freeboard's direct input), and compound_risk (a deterministic
//   function of imerg_24h + tide_range_fes), AND excludes all terrain
//   features (elevation, slope, aspect, dist_water, lt3, lt1, lt0) to
//   be consistent with Model B's terrain-free architecture.
// =====================================================================
var inputFeaturesNoLeak = [
  // [v4.0] Terrain excluded (Model B is terrain-free now)
  "imerg_peak_intensity", "imerg_3d",
  "chirps_7d", "chirps_30d",
  "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
  "soil_moisture",
  "tide_range_fes", "spring_flag_fes",
  "season_wet"
  // EXCLUDED: imerg_24h, freeboard_fes, tide_max_fes, compound_risk,
  // and (never added in the first place) RFI/CII -- all either the
  // literal label-rule inputs (v3.2.14) or deterministic functions of them
];

var rfDriverNoLeak = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).train({
  features: balancedTrainDriver,
  classProperty: "flood_driver",
  inputProperties: inputFeaturesNoLeak
});

print("=== 28b: MODEL B (LEAKAGE-CHECK) FEATURE IMPORTANCE ===");
print(rfDriverNoLeak.explain().get("importance"));

var testDriverNoLeakClassified = testDriver.classify(rfDriverNoLeak, "driver_pred_noleak");
var testDriverNoLeakMatrix = testDriverNoLeakClassified.errorMatrix("flood_driver", "driver_pred_noleak");

print("=== 28b: MODEL B LEAKAGE-CHECK, TEST SET (2023) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2):", testDriverNoLeakMatrix);
print("Overall Accuracy:", testDriverNoLeakMatrix.accuracy());
print("Kappa:", testDriverNoLeakMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", testDriverNoLeakMatrix.producersAccuracy());

var valDriverNoLeakClassified = validationDriver.classify(rfDriverNoLeak, "driver_pred_noleak");
var valDriverNoLeakMatrix = valDriverNoLeakClassified.errorMatrix("flood_driver", "driver_pred_noleak");

print("=== 28b: MODEL B LEAKAGE-CHECK, VALIDATION SET (2024-2025) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2):", valDriverNoLeakMatrix);
print("Overall Accuracy:", valDriverNoLeakMatrix.accuracy());
print("Kappa:", valDriverNoLeakMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", valDriverNoLeakMatrix.producersAccuracy());

print("=== 28b: HOW TO READ THIS ===");
print("Compare Overall Accuracy / Kappa here against section 27/28's");
print("full-feature Model B (test ~0.92/0.86, validation ~0.88/0.82).");
print("Small drop (a few points): genuine independent attribution skill.");
print("Large collapse (toward class-prior baseline): Model B was");
print("substantially reconstructing its own labeling rule, not");
print("attributing flood driver from independent evidence.");

// =====================================================================
// 28c. STRICT LEAKAGE CHECK (v4.0)
//   The partial leakage check (28b) still keeps chirps_30d (an RFI
//   component: Ra), soil_moisture (an RFI component: S), tide_range_fes
//   (a CII component: Tn), and imerg_3d (highly correlated with the
//   excluded imerg_24h). This strict version removes ALL variables that
//   participate in RFI or CII construction, testing with only features
//   that have ZERO role in the label:
//     imerg_peak_intensity -- not in RFI (RFI uses imerg_24h, not peak)
//     chirps_7d            -- not in RFI (RFI uses chirps_30d)
//     solar_rad            -- no role in either index
//     u/v wind, speed, dir -- no role in either index
//     spring_flag_fes      -- related to tides but not in CII formula
//     season_wet           -- no role in either index
//   If accuracy holds with these 9 features: genuine independent
//   attribution from wind/solar/short-term-rain signals.
//   If it collapses: the model was reconstructing labels from their
//   own input variables -- an honest and important result.
// =====================================================================
var inputFeaturesStrictNoLeak = [
  "imerg_peak_intensity",
  "chirps_7d",
  "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
  "spring_flag_fes",
  "season_wet"
];

var rfDriverStrict = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).train({
  features: balancedTrainDriver,
  classProperty: "flood_driver",
  inputProperties: inputFeaturesStrictNoLeak
});

print("=== 28c: STRICT LEAKAGE CHECK -- FEATURE IMPORTANCE ===");
print(rfDriverStrict.explain().get("importance"));

var testDriverStrictClassified = testDriver.classify(rfDriverStrict, "driver_pred_strict");
var testDriverStrictMatrix = testDriverStrictClassified.errorMatrix("flood_driver", "driver_pred_strict");

print("=== 28c: STRICT LEAKAGE CHECK, TEST SET (2023) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2):", testDriverStrictMatrix);
print("Overall Accuracy:", testDriverStrictMatrix.accuracy());
print("Kappa:", testDriverStrictMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", testDriverStrictMatrix.producersAccuracy());

var valDriverStrictClassified = validationDriver.classify(rfDriverStrict, "driver_pred_strict");
var valDriverStrictMatrix = valDriverStrictClassified.errorMatrix("flood_driver", "driver_pred_strict");

print("=== 28c: STRICT LEAKAGE CHECK, VALIDATION SET (2024-2025) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2):", valDriverStrictMatrix);
print("Overall Accuracy:", valDriverStrictMatrix.accuracy());
print("Kappa:", valDriverStrictMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", valDriverStrictMatrix.producersAccuracy());

print("=== 28c: HOW TO READ THIS ===");
print("These 9 features have ZERO role in RFI/CII label construction.");
print("If accuracy holds close to full Model B: genuine independent");
print("attribution from wind, solar, and short-term rain patterns.");
print("If it collapses toward majority-class-guessing: the model was");
print("reconstructing labels from correlated input variables, not");
print("independently attributing flood drivers.");

// =====================================================================
// 29. MODEL B MAP VISUALISATION -- jun2023 driver classification
// =====================================================================
var driverProbMap_jun2023 = vizImg_jun2023.select(inputFeaturesDriverOnly).classify(rfDriver, "driver_pred");
var driverVis = {
  min: 0, max: 2,
  // 0=no-flood(white) 1=rain-dominant(blue) 2=coastal-dominant(orange)
  palette: ["FFFFFF", "1E90FF", "FFA500"]
};
var maskedDriver_jun2023 = driverProbMap_jun2023
  .updateMask(permWater.not())
  .updateMask(driverProbMap_jun2023.gt(0));
Map.addLayer(maskedDriver_jun2023, driverVis, "Model B: Driver classification Jun 2023", false);

var driverLabel_jun2023 = vizImg_jun2023.select("flood_driver")
  .updateMask(permWater.not())
  .updateMask(vizImg_jun2023.select("flood_driver").gt(0));
Map.addLayer(driverLabel_jun2023, driverVis, "Model B: RULE-BASED driver label Jun 2023 (ground truth)", false);

// =====================================================================
// 30. MODEL B EXPORTS
// =====================================================================
Export.table.toDrive({
  collection: samplesDriver,
  description: "keta_samples_driver_v4_0_all_splits",
  fileFormat: "CSV"
});

Export.image.toDrive({
  image: driverProbMap_jun2023.toInt(),
  description: "keta_driver_classification_jun2023_v4_0",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});
