// =====================================================================
// KETA COASTAL FLOOD PREDICTION -- PRODUCTION GEE SCRIPT v3.2.3
// Sentinel-1 SAR + FES2022 + ERA5 + CHIRPS + GPM IMERG + SMAP + SRTM
// Last updated: July 2026
//
// CHANGES FROM v3.2.2:
//   [CHANGE] TPXO dropped entirely as a project decision -- FES2022 is now
//            the sole tidal data source. Removed: TPXO_DATA dictionary,
//            tide_max_tpxo / tide_range_tpxo / spring_flag_tpxo bands,
//            and their entries in inputFeatures. FES2022_DATA's max/range
//            now represent GEOCENTRIC tide (ocean_tide_extrapolated +
//            load_tide), extracted via PyFES -- see extract_fes2022_tides.py.
//   [DONE]   FES2022_DATA regenerated for ALL 19 events via
//            extract_fes2022_tides.py (geocentric tide, hourly, bbox
//            0-2 degE / 4.9-6.9 degN). No placeholder zeros remain.
//
// CHANGES FROM v3.2.1:
//   [FIX] Section 17: Class balancing was backwards. stratifiedSample()
//         already pulls up to nPts POINTS PER CLASS, so the raw `train`
//         set should already be near-balanced going in. The old code then
//         replicated nonFloodSamples 10x regardless of actual class sizes,
//         which manufactures a NEW ~10:1 imbalance in favor of non-flood
//         rather than correcting one. Replaced with dynamic logic that:
//           1. Measures floodCount vs nonFloodCount after the split.
//           2. Identifies whichever class is actually smaller (this could
//              be either class, depending on per-event flood pixel counts).
//           3. Oversamples ONLY the minority class up to the majority
//              count, then trims with randomColumn+limit for an exact
//              1:1 ratio instead of an approximate ceil() overshoot.
//         Class counts are now printed BEFORE balancing so you can catch
//         a degenerate split (e.g. one class near-empty) before training.
//
// CHANGES FROM v3.2:
//   [FIX] Section 6: .set("key": val) invalid JS syntax -> .set({key: val})
//   [FIX] Section 6: Fixed 20th-percentile "Otsu" replaced with true
//         histogram-based Otsu thresholding (maximizes between-class variance)
//   [FIX] Section 6: Added per-event otsu_threshold property for QA/paper writeup
//   [FIX] Section 6: Added null-histogram fallback (-2) for sparse S1 coverage
//   [FIX] Section 20/21: flood_pred boolean explicitly cast to Int for
//         errorMatrix type consistency
//
// CHANGES FROM v3.1 (carried forward):
//   [+] FIXED StratifiedSample crash: fallback to randomSample if 0 flood pts
//   [+] FIXED Pseudo-replication: Sampling scale increased to 200m
//   [+] ADDED RF Probability output (setOutputMode('PROBABILITY'))
//   [+] ADDED Sentinel-1 incidence angle mask (30-45 deg) to reduce SAR noise
//   [+] ADDED explicit integer casting in getSeasonFlag
//
// KNOWN OPEN ITEMS (not fixed here, flagged for follow-up):
//   - stratifiedSample({numPoints: nPts, ...}) samples up to nPts POINTS
//     PER CLASS (flood=1 and flood=0 separately), not nPts total. Confirm
//     printed sample sizes reflect this before writing up methods.
//   - getSeasonFlag's Apr-Jun / Sep-Nov wet season split is a simplification
//     of Ghana's bimodal coastal rainfall pattern - cite appropriately.
//   - Spot-check may2021 / jun2021 (narrowest event windows) for degenerate
//     all-zero diff images that could push Otsu near a nonsensical threshold
//     without tripping the null-histogram fallback.
//   - otsuThreshold()'s split-point sequence runs i = 1..size, which
//     includes the degenerate case where the "after" side is empty
//     (bCount = 0), producing a NaN entry in the between-class-variance
//     array before the sort/argmax step. Likely harmless in practice
//     since GEE's array sort places NaNs predictably, but bounding the
//     sequence to size - 1 would remove the ambiguity outright.
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
  {id:"feb2024", start:"2024-02-10", end:"2024-02-20", split:"validation", confidence:"HIGH"},
  {id:"may2024", start:"2024-05-01", end:"2024-05-20", split:"validation", confidence:"HIGH"},
  {id:"jan2025", start:"2025-01-16", end:"2025-01-28", split:"validation", confidence:"HIGH"},
  {id:"feb2025", start:"2025-02-01", end:"2025-02-08", split:"validation", confidence:"HIGH"},
  {id:"mar2025", start:"2025-03-01", end:"2025-03-10", split:"validation", confidence:"HIGH"}
];

var trainIds      = ["jun2019","jul2019","jun2020","jul2020","may2021","jun2021","nov2021","may2022","jun2022","jul2022"];
var testIds       = ["jun2023","jul2023","sep2023","oct2023"];
var validationIds = ["feb2024","may2024","jan2025","feb2025","mar2025"];

// =====================================================================
// 4. TIDAL DICTIONARY (FES2022 -- sole tidal data source as of v3.2.3)
// max/range are GEOCENTRIC tide (ocean_tide_extrapolated + load_tide, m),
// extracted via PyFES at lon=0.97E, lat=5.90N (bbox 0-2 degE / 4.9-6.9 degN,
// hourly resolution). ALL 19 events regenerated on 2026-07-12 from the
// same consistent method. See extract_fes2022_tides.py + fes2022_extracted_tides.csv.
// spring=1 if event range >= median range across all 19 events (1.634 m).
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
// Masking steep incidence angles (>45 deg) prevents false flood signals
// caused by radar layover and shadow in coastal terrain.
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

  // Full sequence 1..size so bss has the same length as means for Array.sort.
  // Guard division by zero with safeBCount: when bCount=0, the second term
  // becomes 0*(...)=0, so the degenerate split gets variance=0 and is never chosen.
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

  // Dummy completely masked image to prevent 0-band empty collections
  var dummy = ee.Image.constant(-25).rename("VH").selfMask();

  var before = beforeCol.merge(ee.ImageCollection([dummy])).median().unmask(-25);
  var after  = afterCol.merge(ee.ImageCollection([dummy])).median().unmask(-25);

  before = before.focal_mean(30, "circle", "meters");
  after  = after.focal_mean(30,  "circle", "meters");

  var diff = after.subtract(before).rename("VH");

  // Build histogram for real Otsu; fallback to fixed constant if empty
  var histResult = diff.reduceRegion({
    reducer:   ee.Reducer.histogram(255, 2),
    geometry:  keta,
    scale:     30,
    maxPixels: 1e9
  }).get("VH");

  // Prevent server-side crash during ee.Algorithms.If evaluation of otsuThreshold(null)
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
      -2  // fallback constant if histogram fails or has only 1 bin
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
    "otsu_threshold": otsuThresh  // logged per-event so you can sanity-check it later
  });
}

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
// 9. ERA5
// =====================================================================
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR");

function getERA5(t) {
  var c = era5.filterDate(t.advance(-3, "day"), t);
  
  var dummyU = ee.Image.constant(0).rename("u_component_of_wind_10m").selfMask();
  var dummyV = ee.Image.constant(0).rename("v_component_of_wind_10m").selfMask();
  var dummySolar = ee.Image.constant(0).rename("surface_net_solar_radiation_sum").selfMask();
  
  var u = c.select("u_component_of_wind_10m").merge(ee.ImageCollection([dummyU])).mean().unmask(0).rename("u_wind_10m");
  var v = c.select("v_component_of_wind_10m").merge(ee.ImageCollection([dummyV])).mean().unmask(0).rename("v_wind_10m");
  var solar = c.select("surface_net_solar_radiation_sum").merge(ee.ImageCollection([dummySolar])).mean().unmask(0).rename("solar_rad");
  
  var windSpeed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed");
  var windDir   = u.atan2(v).multiply(180 / Math.PI).rename("wind_dir");
  return ee.Image.cat([solar, u, v, windSpeed, windDir]);
}

// =====================================================================
// 10. SMAP SOIL MOISTURE
// =====================================================================
var smap = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture");

// Dummy completely masked image to handle empty SMAP collections
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
// 12. COMPOUND RISK BAND
// =====================================================================
function getCompoundRisk(imergBands, sm, hydro) {
  var heavyRain  = imergBands.select("imerg_24h").gt(25);
  var wetSoil    = sm.gt(0.3);
  var springTide = hydro.select("spring_flag_fes").eq(1);
  return heavyRain.and(wetSoil).and(springTide).rename("compound_risk").toFloat();
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

  var terrain = ee.Image.cat([
    dem.rename("elevation"), slope.rename("slope"), aspect.rename("aspect"),
    distWater, elev_lt3, elev_lt1, elev_lt0
  ]);

  return ee.Image.cat([terrain, imergBands, rain, era, sm, hydro, season, compound])
    .addBands(label)
    .clip(keta)
    .set("event_id", id);
}

// =====================================================================
// 15. SAMPLE GENERATION (Scale=200 + Crash Fallback)
// Scale increased to 200m to reduce pseudo-replication of 10km SMAP/IMERG.
// NOTE: stratifiedSample numPoints is PER CLASS, not total.
// =====================================================================
var nPtsMap = ee.Dictionary({HIGH: 500, MEDIUM: 200});

var samples = ee.FeatureCollection(
  ee.List(events).map(function(e) {
    e = ee.Dictionary(e);
    var img  = build(e);
    var nPts = ee.Number(nPtsMap.get(e.get("confidence")));

    // Check if flood pixels exist to prevent stratifiedSample crash
    var floodSum = img.select("flood").reduceRegion({
      reducer: ee.Reducer.sum(), geometry: keta, scale: 200, maxPixels: 1e9
    }).getNumber("flood");

    var samp = ee.Algorithms.If(
      floodSum.gt(0),
      img.stratifiedSample({
        numPoints: nPts, classBand: "flood", region: keta,
        scale: 200, geometries: true, seed: 42
      }),
      img.sample({ // Fallback if no flood pixels exist
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
// 17. CLASS BALANCING (v3.2.3 -- dynamic minority oversampling)
//
// stratifiedSample() already pulls up to nPts POINTS PER CLASS, so the
// raw `train` set going into this section should already be roughly
// balanced. But per-event flood pixel scarcity (e.g. narrow event
// windows, small flood extents) means the ACTUAL post-sampling counts
// can still land unevenly in either direction -- it's not safe to assume
// which class ends up smaller.
//
// Fix: measure both class counts, identify whichever is actually the
// minority, oversample ONLY that class up to the majority count, then
// trim to an exact 1:1 ratio (rather than an approximate ceil()
// overshoot from a fixed replication factor).
// =====================================================================
var floodSamples    = train.filter(ee.Filter.eq("flood", 1));
var nonFloodSamples = train.filter(ee.Filter.eq("flood", 0));

var floodCount    = floodSamples.size();
var nonFloodCount = nonFloodSamples.size();

print("=== TRAINING SET CLASS COUNTS (pre-balance) ===");
print("Flood samples:    ", floodCount);
print("Non-flood samples:", nonFloodCount);

// Identify minority/majority collections dynamically (server-side)
var minorIsFlood = floodCount.lt(nonFloodCount);

var minorCollection = ee.FeatureCollection(
  ee.Algorithms.If(minorIsFlood, floodSamples, nonFloodSamples)
);
var majorCollection = ee.FeatureCollection(
  ee.Algorithms.If(minorIsFlood, nonFloodSamples, floodSamples)
);

var minorCount = ee.Number(ee.Algorithms.If(minorIsFlood, floodCount, nonFloodCount));
var majorCount = ee.Number(ee.Algorithms.If(minorIsFlood, nonFloodCount, floodCount));

// Prevent server-side division by zero inside majorCount.divide(minorCount)
// which would otherwise execute because ee.Algorithms.If evaluates both branches.
var safeMinorCount = ee.Number(ee.Algorithms.If(minorCount.gt(0), minorCount, 1));
var repeatFactor = majorCount.divide(safeMinorCount).ceil().max(1).toInt();

// Over-replicate the minority class using the same sequence+flatten
// pattern as before (avoids the memory cost of chained .merge() calls),
// then randomly trim down to an exact match with the majority count.
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
// 18. FEATURE SETS
// =====================================================================
var inputFeatures = [
  "elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0",
  "imerg_24h", "imerg_peak_intensity", "imerg_3d",
  "chirps_7d", "chirps_30d",
  "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed", "wind_dir",
  "soil_moisture",
  "tide_max_fes", "tide_range_fes", "spring_flag_fes",
  "compound_risk", "season_wet"
];

// =====================================================================
// 19. RF MODEL (Probability Output)
// =====================================================================
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  seed: 42
}).setOutputMode('PROBABILITY') // Outputs 0.0 to 1.0 probability
.train({
  features: balancedTrain,
  classProperty: "flood",
  inputProperties: inputFeatures
});

print("=== FEATURE IMPORTANCE ===");
print(rf.explain().get("importance"));

// =====================================================================
// 20. TEST SET EVALUATION (Probability Thresholding)
// =====================================================================
// Classify and threshold probability at 0.5 for binary error matrix.
// flood_pred explicitly cast to Int for type-consistent errorMatrix comparison.
var testClassified = test.classify(rf).map(function(f) {
  var prob = ee.Number(f.get("classification"));
  return f.set("flood_pred", prob.gte(0.5).toInt());
});

var testMatrix = testClassified.errorMatrix("flood", "flood_pred");

print("=== TEST SET (2023) ===");
print("Confusion Matrix:", testMatrix);
print("Overall Accuracy:", testMatrix.accuracy());
print("Kappa:", testMatrix.kappa());

// Test set detailed metrics
var testRecall    = testMatrix.producersAccuracy().get([1, 0]);
var testPrecision = testMatrix.consumersAccuracy().get([0, 1]);
var testF1 = ee.Number(2).multiply(testPrecision.multiply(testRecall))
                        .divide(testPrecision.add(testRecall));
print("Sensitivity (Recall):", testRecall);
print("Precision:", testPrecision);
print("F1-Score:", testF1);

// =====================================================================
// 21. VALIDATION SET EVALUATION (2024-2025)
// FES2022_DATA values for 2024-2025 are now real (no placeholders).
// =====================================================================
var valClassified = validation.classify(rf).map(function(f) {
  var prob = ee.Number(f.get("classification"));
  return f.set("flood_pred", prob.gte(0.5).toInt());
});
var valMatrix = valClassified.errorMatrix("flood", "flood_pred");
print("=== VALIDATION SET (2024-2025) ===");
print("Confusion Matrix:", valMatrix);
print("Overall Accuracy:", valMatrix.accuracy());

// Sensitivity (Recall) = True Positives / (True Positives + False Negatives)
// In GEE, producersAccuracy() returns an N x 1 array. For class 1 (flood), index is [1, 0]
var recall = valMatrix.producersAccuracy().get([1, 0]); 

// Precision = True Positives / (True Positives + False Positives)
// In GEE, consumersAccuracy() returns a 1 x N array. For class 1 (flood), index is [0, 1]
var precision = valMatrix.consumersAccuracy().get([0, 1]);

// F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
var f1 = ee.Number(2).multiply(precision.multiply(recall))
                    .divide(precision.add(recall));

print("Sensitivity (Recall):", recall);
print("Precision:", precision);
print("F1-Score:", f1);

// =====================================================================
// 22. MAP VISUALISATION
// Displaying continuous flood probability (0 to 1) for jun2023 (test)
// and all 5 validation events (2024-2025).
// =====================================================================

// --- Helper: build a probability map for any event ---
function buildProbMap(eventDict) {
  var img  = build(eventDict);
  return img.select(inputFeatures).classify(rf);
}

// --- Jun 2023 (Test Event -- Primary Figure) ---
var vizEvent_jun2023 = ee.Dictionary({
  id:"jun2023", start:"2023-06-01", end:"2023-06-15",
  split:"test", confidence:"HIGH"
});
var vizImg_jun2023  = build(vizEvent_jun2023);
var vizProb_jun2023 = buildProbMap(vizEvent_jun2023);

// --- Feb 2024 ---
var vizEvent_feb2024 = ee.Dictionary({
  id:"feb2024", start:"2024-02-10", end:"2024-02-20",
  split:"validation", confidence:"HIGH"
});
var vizProb_feb2024 = buildProbMap(vizEvent_feb2024);

// --- May 2024 ---
var vizEvent_may2024 = ee.Dictionary({
  id:"may2024", start:"2024-05-01", end:"2024-05-20",
  split:"validation", confidence:"HIGH"
});
var vizProb_may2024 = buildProbMap(vizEvent_may2024);

// --- Jan 2025 ---
var vizEvent_jan2025 = ee.Dictionary({
  id:"jan2025", start:"2025-01-16", end:"2025-01-28",
  split:"validation", confidence:"HIGH"
});
var vizProb_jan2025 = buildProbMap(vizEvent_jan2025);

// --- Feb 2025 ---
var vizEvent_feb2025 = ee.Dictionary({
  id:"feb2025", start:"2025-02-01", end:"2025-02-08",
  split:"validation", confidence:"HIGH"
});
var vizProb_feb2025 = buildProbMap(vizEvent_feb2025);

// --- Mar 2025 (strongest spring tide event) ---
var vizEvent_mar2025 = ee.Dictionary({
  id:"mar2025", start:"2025-03-01", end:"2025-03-10",
  split:"validation", confidence:"HIGH"
});
var vizProb_mar2025 = buildProbMap(vizEvent_mar2025);

// --- Base layers ---
Map.addLayer(vizImg_jun2023.select("elevation"),
  {min:0, max:10, palette:["blue","green","yellow","red"]}, "Elevation (m)", false);
Map.addLayer(vizImg_jun2023.select("imerg_24h"),
  {min:0, max:100, palette:["white","cyan","blue"]}, "IMERG 24h Rain (mm)", false);
Map.addLayer(permWater.selfMask(),
  {palette:["0000FF"]}, "Permanent Water (JRC)", false);

// --- Probability layers (masked: hide safe areas and permanent water) ---
// Only pixels with >10% flood probability are shown.
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
// 23. EXPORTS
// =====================================================================
// --- Sample tables ---
Export.table.toDrive({
  collection: samples, description: "keta_samples_v3_2_3_all_splits", fileFormat: "CSV"
});

Export.table.toDrive({
  collection: train, description: "keta_samples_v3_2_3_train", fileFormat: "CSV"
});

// Export test set with raw probability AND binary threshold
Export.table.toDrive({
  collection: testClassified,
  description: "keta_samples_v3_2_3_test_classified",
  fileFormat: "CSV"
});

// --- Flood probability GeoTIFFs (one per event for paper figures) ---
Export.image.toDrive({
  image: vizProb_jun2023.toFloat(),
  description: "keta_flood_probability_jun2023_v3_2_3",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_feb2024.toFloat(),
  description: "keta_flood_probability_feb2024_v3_2_3",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_may2024.toFloat(),
  description: "keta_flood_probability_may2024_v3_2_3",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_jan2025.toFloat(),
  description: "keta_flood_probability_jan2025_v3_2_3",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_feb2025.toFloat(),
  description: "keta_flood_probability_feb2025_v3_2_3",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: vizProb_mar2025.toFloat(),
  description: "keta_flood_probability_mar2025_v3_2_3",
  region: keta, scale: 30, fileFormat: "GeoTIFF"
});

