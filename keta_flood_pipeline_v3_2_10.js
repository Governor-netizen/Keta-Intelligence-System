// =====================================================================
// KETA COASTAL FLOOD PREDICTION -- PRODUCTION GEE SCRIPT v3.2.10
// Sentinel-1 SAR + FES2022 + ERA5 + CHIRPS + GPM IMERG + SMAP + SRTM
// Last updated: July 2026
//
// CHANGES FROM v3.2.9:
//   [DIAG] Section 24c: imerg_24h mean/stdDev/min/max per event (all 19),
//         added to check whether heavyRain (imerg_24h>25mm) is behaving
//         as a real pixel-level spatial feature or an event-level near-
//         constant -- same question the freeboard fix (v3.2.7) answered
//         for tide. Motivated by test/validation confusion matrices
//         coming back 3x3 instead of 4x4 (v3.2.9 run): classes 1
//         (rain-driven) and 3 (compound) had ZERO actual samples in
//         both test and validation, despite existing in training (67
//         and 122 samples respectively). If imerg_24h rarely/never
//         crosses 25mm within a given event's samples, no pixel in that
//         event can ever land in class 1 or 3, regardless of true
//         driver -- which would explain the disappearance directly.
//   [CHANGE] Section 13b/getDriverLabel: added class 4 = "ambiguous"
//         for flood==1 pixels matching NEITHER heavy-rain NOR tidal-
//         inundation. Previously (v3.2.7-v3.2.9) these silently fell
//         into class 2 (surge-driven) via a catch-all default -- this
//         was flagged in the original v3.2.8 comment as "a judgment
//         call, worth revisiting once the actual class distribution is
//         visible." It's now visible: class 2 held 91% of all flood
//         samples (1946/2135) pre-balance, and the 24b-Q1 diagnostic
//         showed flood==1 freeboard mean (0.686) LOWER than flood==0
//         (0.859) with 2x the stdDev -- consistent with true tidal
//         pixels (freeboard>0) being diluted by mislabeled ambiguous
//         ones (freeboard could be anything) inside the same class.
//         Splitting them out stops that contamination. Model B is now
//         5-class (0-4); driverClasses, balancing, confusion matrix,
//         and driverVis palette updated accordingly. Every downstream
//         section that iterates driverClasses picks this up
//         automatically since none of them hardcode the class count.
//   [NOTE] This changes what "surge-driven" means retroactively -- if
//         you're comparing class-2 counts/results against anything
//         computed under v3.2.7-v3.2.9, that comparison is no longer
//         apples-to-apples. Worth a footnote if this ends up in the
//         paper writeup.
//
// CHANGES FROM v3.2.8:
//   [ADD] Section 24b: freeboard/DEM sanity diagnostic. Added after the
//         Freeboard (tide_max - elevation) map layer showed a single
//         uniform orange band (~-0.33 to +0.83m) across almost the
//         entire mainland, raising two open questions this section
//         answers with numbers instead of eyeballing the map:
//           Q1: does freeboard_fes actually separate flood==1 from
//               flood==0 samples (mean/stdDev by group), or is the RF
//               getting its signal from elevation/dist_water directly
//               and freeboard isn't pulling weight?
//           Q2: is the flat mainland orange real low-relief geography,
//               or SRTM vertical noise (~+-5-10m in flat coastal terrain)
//               flattening real elevation differences into a narrow band?
//               Checked via elevation percentile/stdDev spread split by
//               dist_water (<=100m as spit/lagoon-edge proxy, >=500m as
//               mainland-interior proxy) plus a whole-ROI histogram.
//         Read-only diagnostic -- runs off samplesDriver (already built
//         in section 24) and distWater (already built in section 2), no
//         new bands or exports added. Nothing downstream changed.
//
// CHANGES FROM v3.2.7:
//   [ADD] MODEL B: compound flood DRIVER ATTRIBUTION, added as entirely
//         new sections (13b, 24-30) alongside the existing binary
//         susceptibility classifier (Model A, sections 1-23, UNCHANGED).
//         Two separate, honestly-scoped deliverables instead of one
//         model asked to do both jobs:
//           Model A = WHERE is flood-prone (binary, terrain-dominated,
//                     already validated ~90-93% test/val accuracy).
//           Model B = WHY did a flooded pixel flood (4-class: no-flood /
//                     rain-driven / surge-driven / compound).
//         Model B's label (flood_driver, section 13b) is RULE-BASED, not
//         derived from the SAR signal -- SAR only detects THAT a pixel
//         flooded, not why. Rule: heavy rain (imerg_24h>25mm, event-
//         level) x tidal inundation (freeboard_fes>0, pixel-level) ->
//         4-way split. Flood pixels matching neither condition default
//         to surge-driven rather than a 5th "ambiguous" class -- a
//         judgment call, documented in section 13b, worth revisiting
//         once the actual class distribution is visible.
//         Model B required its OWN stratified sampling (section 24) --
//         Model A's `samples` was stratified on binary flood/no-flood
//         only, so driver-class balance among flood==1 pixels was never
//         controlled for and can't be reused -- plus its own N-class
//         balancing (section 25, generalized from Model A's binary-only
//         version) and its own 4x4 confusion matrix evaluation (27-28).
//   [ACTION REQUIRED] Check the printed pre-balance class counts (section
//         24) before trusting Model B's results -- with only 10 training
//         events, it's plausible one or more driver classes end up
//         severely underrepresented or entirely absent, which the code
//         detects and skips rather than fabricates, but that still means
//         Model B may not be reliable for that class until more events
//         are added or the labeling thresholds are revisited.
//
//   [FIX 2026-07-13] Section 25: balancedTrainDriver crashed with
//         "Unrecognized argument type to convert to a FeatureCollection:
//         null" on the "Balanced total:" print. Root cause: the merge
//         reduce started its accumulator at a real JS `null` and used
//         `ee.Algorithms.If(acc === null, part, ee.FeatureCollection(acc)
//         .merge(...))` to special-case the first iteration -- but
//         ee.Algorithms.If evaluates BOTH branches eagerly as part of
//         building the computation graph, regardless of which one is
//         logically selected. So even on the first iteration (where the
//         "merge" branch should never be used), GEE still tried to build
//         `ee.FeatureCollection(acc)` with acc literally equal to null,
//         and that conversion is what threw. Same eager-both-branches
//         trap this pipeline already guards against elsewhere (the
//         divide-by-zero guards in section 17, the tide-constant
//         fallback in section 11). Fixed by starting the reduce's
//         accumulator at a real ee.FeatureCollection([]) instead of JS
//         null, and dropping the acc === null special case entirely --
//         once the accumulator is never really null, every iteration can
//         just merge unconditionally.
//
// CHANGES FROM v3.2.6:
//   [ADD] Section 11b: new freeboard_fes band (tide_max_fes - elevation),
//         per-pixel, added to inputFeatures. Fixes the structural reason
//         tide_max_fes/tide_range_fes ranked low in feature importance --
//         they're flat per-event constants with almost no split
//         opportunities for Gini-based RF importance, regardless of true
//         causal weight (per-event dist_water diagnostic run 2026-07-13:
//         flood samples spanned 0-1173m from water, confirming terrain
//         features vary meaningfully where tide constants can't).
//   [CHANGE] Section 12: compound_risk rebuilt as a continuous 0-1 PRODUCT
//         of normalized rain/soil-moisture/tide-range, replacing the old
//         hard AND (imerg_24h>25 AND soil_moisture>0.3 AND
//         spring_flag_fes==1). The old version scored exactly 0.000
//         importance -- confirmed via feature importance diagnostic --
//         almost certainly because the three-way AND was true for near
//         zero training samples. Kept as a product (not sum/average) to
//         preserve the "needs multiple simultaneous drivers" intent.
//   [NOTE] Both changes are feature-level only -- the flood LABEL is
//         still binary (flood/no-flood). Driver attribution (rain- vs
//         surge- vs compound-driven) is NOT yet implemented; that would
//         require rebuilding getFloodLabel() into a multiclass target
//         and is a larger, separate change, deferred per project
//         decision on 2026-07-13.
//   [KNOWN GAP, not addressed here] Storm surge itself is still not
//         represented anywhere in the feature stack. FES2022 is a pure
//         astronomical + load tide model; wind_speed/wind_dir are the
//         closest available proxy but are not a real surge estimate.
//         A proper fix would need a surge/reanalysis product (e.g. GTSM)
//         not currently in this pipeline.
//
// CHANGES FROM v3.2.5:
//   [FIX] Section 3/22: widened feb2024 (end 2024-02-20 -> 2024-03-05) and
//         mar2025 (end 2025-03-10 -> 2025-03-25) event windows. The new
//         6b Sentinel-1 coverage diagnostic (added in v3.2.5) confirmed
//         both events had after_images=0 -- zero real Sentinel-1 passes
//         in the post-event window -- while before_images=1 (real). This
//         is a quieter failure than a fully-empty before+after pair: since
//         `before` is real and `after` is the dummy -25 constant, the diff
//         image still has real spatial texture, so Otsu runs on a normal-
//         looking histogram and produces a threshold that LOOKS legitimate
//         but isn't measuring any actual post-event radar signal. Neither
//         the histogram-empty fallback nor any existing check catches this.
//         mar2025 is notable because it's flagged in this script's own
//         comments as the headline "strongest spring tide" validation map.
//         14 of the other 17 events already have real coverage in both
//         windows; most sit at only 1-2 images per window (not multi-look
//         averaging in any meaningful sense) -- worth stating plainly in
//         the methods writeup rather than implying denser compositing
//         than what's actually happening.
//
// CHANGES FROM v3.2.4:
//   [FIX] Section 9: ERA5/DAILY replaced with ERA5/HOURLY (aggregated to
//         a 3-day mean/sum). ERA5/DAILY's Earth Engine catalog listing
//         stopped being extended after 2020-07-09 -- it returned ZERO
//         images for every event in this study from 2021 onward.
//         ERA5/HOURLY is documented as available 1940-present, non-
//         land-masked, with identical variable/band-naming convention.
//   [RESTORED] solar_rad from ERA5/HOURLY's surface_net_solar_radiation.
//
// CHANGES FROM v3.2.3:
//   [FIX] Section 9: ERA5_LAND/DAILY_AGGR replaced with ERA5/DAILY.
//         ERA5_LAND (11km, land-masked) had ZERO valid pixels over the
//         Keta ROI. [SUPERSEDED in v3.2.5 -- ERA5/DAILY turned out to
//         have stopped updating in 2020, a separate bug.]
//   [FIX] Section 14: build()'s final .set('event_id') was missing its
//         value argument. Restored to .set('event_id', id).
//
// CHANGES FROM v3.2.2:
//   [CHANGE] TPXO dropped entirely -- FES2022 is now the sole tidal data
//            source, GEOCENTRIC tide extracted via PyFES.
//
// CHANGES FROM v3.2.1:
//   [FIX] Section 17: Class balancing was backwards -- fixed with dynamic
//         minority-oversampling logic that measures actual class sizes
//         and oversamples only the true minority up to the majority count.
//
// CHANGES FROM v3.2:
//   [FIX] Section 6: .set("key": val) invalid JS syntax -> .set({key: val})
//   [FIX] Section 6: true histogram-based Otsu thresholding
//   [FIX] Section 20/21: flood_pred boolean explicitly cast to Int
//
// CHANGES FROM v3.1 (carried forward):
//   [+] FIXED StratifiedSample crash: fallback to randomSample if 0 flood pts
//   [+] FIXED Pseudo-replication: Sampling scale increased to 200m
//   [+] ADDED RF Probability output (setOutputMode('PROBABILITY'))
//   [+] ADDED Sentinel-1 incidence angle mask (30-45 deg)
//   [+] ADDED explicit integer casting in getSeasonFlag
//
// KNOWN OPEN ITEMS (not fixed here, flagged for follow-up):
//   - stratifiedSample({numPoints: nPts, ...}) samples up to nPts POINTS
//     PER CLASS, not nPts total. Confirm printed sample sizes reflect this.
//   - getSeasonFlag's Apr-Jun / Sep-Nov wet season split is a simplification
//     of Ghana's bimodal coastal rainfall pattern - cite appropriately.
//   - otsuThreshold()'s split-point sequence runs i = 1..size, including
//     the degenerate bCount=0 case -- likely harmless given GEE's array
//     sort behavior, but bounding to size-1 would remove the ambiguity.
//   - ERA5/HOURLY is ~28-31km resolution vs. your 200m sampling scale --
//     wind and solar values are near-uniform across the whole ROI within
//     one event. freeboard_fes (v3.2.7) addresses this for tide; wind/
//     solar remain event-level-uniform.
//   - Aggregating ERA5/HOURLY with a straight .mean() over the 3-day
//     window discards intra-event timing -- a wind gust concentrated in
//     6 hours looks identical to the same total spread over 3 days.
//   - Model B (v3.2.8): flood_driver is RULE-BASED (imerg_24h>25mm x
//     freeboard_fes>0), not derived from SAR itself -- describe as a
//     modeling assumption in any writeup, not a measurement. Storm surge
//     is not separately represented; FES2022 is pure astronomical tide.
//     Ambiguous flood pixels (neither condition triggers) default to
//     surge-driven rather than a 5th class -- revisit if the class
//     distribution (see section 24 output) doesn't support that.
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
  {id:"mar2025", start:"2025-03-01", end:"2025-03-25", split:"validation", confidence:"HIGH"} // [WIDENED 2026-07-13]
];

var trainIds      = ["jun2019","jul2019","jun2020","jul2020","may2021","jun2021","nov2021","may2022","jun2022","jul2022"];
var testIds       = ["jun2023","jul2023","sep2023","oct2023"];
var validationIds = ["feb2024","may2024","jan2025","feb2025","mar2025"];

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
  "mar2025":{ max:0.914, range:1.749, spring:1 }
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

  var beforeCol = s1.filterDate(t0.advance(-12, "day"), t0);
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

function getIMERG(t) {
  var dummy = ee.Image.constant(0).rename("precipitation").selfMask();

  var col24h = imerg.filterDate(t.advance(-1, "day"), t).select("precipitation").merge(ee.ImageCollection([dummy]));
  var col3d = imerg.filterDate(t.advance(-3, "day"), t).select("precipitation").merge(ee.ImageCollection([dummy]));

  var imerg24h  = col24h.sum().multiply(0.5).unmask(0).rename("imerg_24h");
  var imergPeak = col3d.max().unmask(0).rename("imerg_peak_intensity");
  var imerg3d   = col3d.sum().multiply(0.5).unmask(0).rename("imerg_3d");
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
// 13b. DRIVER ATTRIBUTION LABEL (Model B -- 2026-07-13, revised v3.2.10)
//   0 = no flood        (flood == 0)
//   1 = rain-driven      (flood==1, heavy rain, NOT tidal inundation)
//   2 = surge-driven      (flood==1, tidal inundation, NOT heavy rain)
//   3 = compound         (flood==1, BOTH heavy rain AND tidal inundation)
//   4 = ambiguous        (flood==1, NEITHER heavy rain NOR tidal
//                          inundation -- v3.2.9's diagnostic showed this
//                          was 91% of all flood samples when it was
//                          silently folded into class 2; broken out as
//                          its own class rather than guessed at)
// =====================================================================
function getDriverLabel(floodBand, imergBands, freeboardBand) {
  var heavyRain        = imergBands.select("imerg_24h").gt(25);
  var tidalInundation  = freeboardBand.gt(0);
  var isFlood          = floodBand.eq(1);

  var driver = ee.Image(0)
    .where(isFlood.and(heavyRain).and(tidalInundation), 3)
    .where(isFlood.and(heavyRain).and(tidalInundation.not()), 1)
    .where(isFlood.and(heavyRain.not()).and(tidalInundation), 2)
    .where(isFlood.and(heavyRain.not()).and(tidalInundation.not()), 4)
    .rename("flood_driver")
    .toInt();

  return driver;
}

// =====================================================================
// 14. FEATURE STACK BUILDER
// =====================================================================
function build(event) {
  var t0         = ee.Date(event.get("start"));
  var id         = event.get("id");
  var imergBands = getIMERG(t0);
  var rain       = getRain(t0);
  var era        = getERA5(t0);
  var sm         = getSM(t0);
  var hydro      = getHydroBands(id);
  var label      = getFloodLabel(event);
  var season     = getSeasonFlag(t0);
  var compound   = getCompoundRisk(imergBands, sm, hydro);
  var freeboard  = getFreeboard(hydro, dem);
  var driverLabel = getDriverLabel(label, imergBands, freeboard);

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
// 23. EXPORTS (Model A)
// =====================================================================
Export.table.toDrive({
  collection: samples,
  description: "keta_samples_v3_2_10_all_splits",
  fileFormat: "CSV"
});

Export.image.toDrive({
  image: vizProb_jun2023.toFloat(),
  description: "keta_flood_probability_jun2023_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_feb2024.toFloat(),
  description: "keta_flood_probability_feb2024_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_may2024.toFloat(),
  description: "keta_flood_probability_may2024_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_jan2025.toFloat(),
  description: "keta_flood_probability_jan2025_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_feb2025.toFloat(),
  description: "keta_flood_probability_feb2025_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_mar2025.toFloat(),
  description: "keta_flood_probability_mar2025_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

// =====================================================================
// =====================================================================
// MODEL B: COMPOUND FLOOD DRIVER ATTRIBUTION (v3.2.8 -- 2026-07-13)
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
print("0 = no-flood:     ", trainDriver.filter(ee.Filter.eq("flood_driver", 0)).size());
print("1 = rain-driven:  ", trainDriver.filter(ee.Filter.eq("flood_driver", 1)).size());
print("2 = surge-driven: ", trainDriver.filter(ee.Filter.eq("flood_driver", 2)).size());
print("3 = compound:     ", trainDriver.filter(ee.Filter.eq("flood_driver", 3)).size());
print("4 = ambiguous:    ", trainDriver.filter(ee.Filter.eq("flood_driver", 4)).size());

// =====================================================================
// 24c. IMERG_24H SPATIAL-VS-EVENT-CONSTANT DIAGNOSTIC (2026-07-13)
//   Same question the freeboard fix answered for tide: is imerg_24h a
//   real pixel-level spatial feature, or a near-constant per event that
//   just happens to vary event-to-event? Motivated by classes 1/3
//   (rain-driven/compound) having ZERO actual samples in test+validation
//   in the v3.2.9 run despite existing in training -- if an event's
//   imerg_24h rarely/never crosses 25mm anywhere in the ROI, no pixel in
//   that event can ever be labeled class 1 or 3, regardless of the true
//   driver. Read-only, all 19 events, uses samplesDriver (section 24).
// =====================================================================
var allEventIds = trainIds.concat(testIds).concat(validationIds);

var imergByEvent = ee.FeatureCollection(allEventIds.map(function(id) {
  var evtSamples = samplesDriver.filter(ee.Filter.eq("event_id", id));
  var stats = evtSamples.reduceColumns({
    reducer: ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true}),
    selectors: ["imerg_24h"]
  });
  return ee.Feature(null, {
    event_id: id,
    n: evtSamples.size(),
    imerg_mean: stats.get("mean"),
    imerg_stdDev: stats.get("stdDev"),
    imerg_min: stats.get("min"),
    imerg_max: stats.get("max")
  });
}));

print("=== 24c: imerg_24h mean/stdDev/min/max per event, all 19 events ===", imergByEvent);
print("If imerg_max stays under 25 for an event, NO pixel in that event");
print("can ever be labeled class 1 (rain-driven) or 3 (compound), no");
print("matter what actually happened on the ground -- that's an event-");
print("selection artifact of the 25mm threshold, not a spatial signal.");
print("If stdDev is small relative to mean within most events, imerg_24h");
print("is behaving as an event-level constant (like tide_max_fes did");
print("before the freeboard fix), not a real per-pixel spatial feature.");

// =====================================================================
// 24b. FREEBOARD / DEM SANITY DIAGNOSTIC (2026-07-13)
//   Q1: does freeboard_fes actually separate flood==1 from flood==0?
//   Q2: is the flat orange mainland real low-relief terrain, or SRTM
//       vertical noise flattening real elevation differences?
//   Read-only -- uses samplesDriver (section 24) and distWater
//   (section 2). No new bands, no exports, nothing downstream changed.
// =====================================================================

// --- Q1: freeboard stats by flood label, using the sample points
// already exported in samplesDriver (has flood, freeboard_fes,
// elevation, dist_water as properties) ---
var freeboardByFlood = samplesDriver.reduceColumns({
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }).group({
    groupField: 1,
    groupName: "flood"
  }),
  selectors: ["freeboard_fes", "flood"]
});
print("=== 24b-Q1: Freeboard mean/stdDev, grouped by flood (0/1) ===", freeboardByFlood);
// If flood==1 mean isn't meaningfully higher (less negative) than
// flood==0, freeboard isn't actually discriminating -- it's just
// correlated with dist_water/elevation, which the RF could get more
// directly from those bands already in inputFeatures.

// --- Q2a: elevation spread over the WHOLE roi (sanity baseline) ---
var elevStatsAll = dem.clip(keta).reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95]).combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }),
  geometry: keta, scale: 30, maxPixels: 1e9
});
print("=== 24b-Q2a: Elevation percentiles + stdDev, whole ROI (m) ===", elevStatsAll);

// --- Q2b: split by distance-to-water as a mainland/spit proxy.
// Close to water (<=100m) approximates the barrier spit / lagoon edge;
// far from water (>=500m) approximates mainland interior. ---
var nearWaterMask = distWater.lte(100);
var farWaterMask  = distWater.gte(500);

var elevNearWater = dem.updateMask(nearWaterMask).clip(keta).reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95]).combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }),
  geometry: keta, scale: 30, maxPixels: 1e9
});
print("=== 24b-Q2b: Elevation percentiles + stdDev, <=100m from water (m) ===", elevNearWater);

var elevFarWater = dem.updateMask(farWaterMask).clip(keta).reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95]).combine({
    reducer2: ee.Reducer.stdDev(), sharedInputs: true
  }),
  geometry: keta, scale: 30, maxPixels: 1e9
});
print("=== 24b-Q2b: Elevation percentiles + stdDev, >=500m from water (m) ===", elevFarWater);

// --- Q2c: visual histogram, whole-ROI elevation, for eyeballing shape ---
print(ui.Chart.image.histogram({
  image: dem.clip(keta), region: keta, scale: 30, maxBuckets: 40
}).setOptions({title: "24b-Q2c: Elevation histogram, whole ROI"}));

// How to read this:
//  Q1 -- if flood==1's freeboard mean isn't clearly less-negative than
//        flood==0's, freeboard_fes isn't earning its place as a feature.
//  Q2 -- if elevFarWater's spread is nearly as narrow as elevNearWater's
//        (e.g. both squeezed into ~0-2m), that points to SRTM flattening
//        real mainland relief -- a DEM accuracy red flag, not geography.
//        If elevFarWater DOES show real spread but the map still renders
//        mostly orange, it's the freeboard palette bucket width
//        (~1.17m/color over the current -5..2 range) that's too coarse
//        to show it -- a display fix (tighten min/max), not a data bug.

// =====================================================================
// 25. N-CLASS BALANCING (generalized from Model A's 2-class version)
//
// [FIX 2026-07-13] Rewrote the final merge to avoid ever handing a real
// JS `null` to ee.FeatureCollection(). Previously the reduce started
// with `null` as its accumulator and used ee.Algorithms.If(acc === null,
// ...) to special-case the first step -- but ee.Algorithms.If builds
// BOTH branches into the computation graph regardless of which one is
// logically chosen, so `ee.FeatureCollection(acc)` was still attempted
// with acc===null on the first iteration, throwing "Unrecognized
// argument type to convert to a FeatureCollection: null". Fixed by
// starting the reduce at a real, empty ee.FeatureCollection([]) instead
// -- every iteration can then just merge unconditionally, with no null
// ever entering the graph.
// =====================================================================
var driverClasses = [0, 1, 2, 3, 4]; // [CHANGE v3.2.10] added class 4 = ambiguous

var driverCollectionsByClass = driverClasses.map(function(c) {
  return trainDriver.filter(ee.Filter.eq("flood_driver", c));
});
var driverCountsByClass = driverCollectionsByClass.map(function(fc) {
  return fc.size();
});

var maxDriverCount = ee.Number(driverCountsByClass[0])
  .max(driverCountsByClass[1]).max(driverCountsByClass[2]).max(driverCountsByClass[3])
  .max(driverCountsByClass[4]);

var balancedDriverParts = driverClasses.map(function(c, i) {
  var coll  = driverCollectionsByClass[i];
  var count = ee.Number(driverCountsByClass[i]);
  var safeCount = ee.Number(ee.Algorithms.If(count.gt(0), count, 1));
  var repeatFactor = maxDriverCount.divide(safeCount).ceil().max(1).toInt();

  var repeated = ee.FeatureCollection(
    ee.List.sequence(1, repeatFactor).map(function(i) { return coll; })
  ).flatten();

  // [FIX 2026-07-13] Always return a real FeatureCollection (empty if the
  // class is absent) rather than a raw `null` -- null is only safe to
  // return from ee.Algorithms.If if NOTHING downstream ever tries to
  // wrap it in ee.FeatureCollection(...), and section-25's merge step
  // used to do exactly that.
  return ee.Algorithms.If(
    count.gt(0),
    repeated.randomColumn("rand_balance_driver", 42).sort("rand_balance_driver").limit(maxDriverCount),
    ee.FeatureCollection([]) // class absent from training data -- empty, not null
  );
});

// [FIX 2026-07-13] Reduce now starts at a REAL empty FeatureCollection,
// not JS null -- so every iteration can merge unconditionally with no
// ee.Algorithms.If / null special-casing needed at all.
var balancedTrainDriver = driverClasses.reduce(function(acc, c, i) {
  var part = ee.FeatureCollection(balancedDriverParts[i]);
  return acc.merge(part);
}, ee.FeatureCollection([]));

print("=== MODEL B: TRAINING SET (post-balance) ===");
print("Balanced total:", balancedTrainDriver.size());
print("NOTE: if any class printed 0 above (section 24), it was entirely");
print("absent from your 10 training events -- check whether the driver-");
print("labeling thresholds (25mm rain, freeboard>0) are realistic for this");
print("dataset before trusting Model B's results for that class.");

// =====================================================================
// 26. MODEL B RF (multiclass, default classification mode)
// =====================================================================
var rfDriver = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).train({
  features: balancedTrainDriver,
  classProperty: "flood_driver",
  inputProperties: inputFeatures
});

print("=== MODEL B: FEATURE IMPORTANCE ===");
print(rfDriver.explain().get("importance"));

// =====================================================================
// 27. MODEL B TEST SET EVALUATION (5x5 confusion matrix)
// =====================================================================
var testDriverClassified = testDriver.classify(rfDriver, "driver_pred");
var testDriverMatrix = testDriverClassified.errorMatrix("flood_driver", "driver_pred");

print("=== MODEL B TEST SET (2023) ===");
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2/3/4):", testDriverMatrix);
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
print("Confusion Matrix (rows=actual, cols=predicted, order 0/1/2/3/4):", valDriverMatrix);
print("Overall Accuracy:", valDriverMatrix.accuracy());
print("Kappa:", valDriverMatrix.kappa());
print("Per-class Producer's Accuracy (recall):", valDriverMatrix.producersAccuracy());
print("Per-class Consumer's Accuracy (precision):", valDriverMatrix.consumersAccuracy());

// =====================================================================
// 29. MODEL B MAP VISUALISATION -- jun2023 driver classification
// =====================================================================
var driverProbMap_jun2023 = vizImg_jun2023.select(inputFeatures).classify(rfDriver, "driver_pred");
var driverVis = {
  min: 0, max: 4,
  // 0=no-flood(white) 1=rain-driven(blue) 2=surge-driven(orange)
  // 3=compound(purple) 4=ambiguous(gray) [CHANGE v3.2.10: added class 4]
  palette: ["FFFFFF", "1E90FF", "FFA500", "8B008B", "808080"]
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
  description: "keta_samples_driver_v3_2_10_all_splits",
  fileFormat: "CSV"
});

Export.image.toDrive({
  image: driverProbMap_jun2023.toInt(),
  description: "keta_driver_classification_jun2023_v3_2_10",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});
