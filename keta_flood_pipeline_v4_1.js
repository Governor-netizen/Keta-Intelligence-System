// =====================================================================
// KETA COASTAL FLOOD PREDICTION -- PRODUCTION GEE SCRIPT v4.1
// Sentinel-1 SAR + FES2022 + ERA5 + CHIRPS + GPM IMERG + SMAP + SRTM
// Last updated: July 2026
//
// CHANGES FROM v4.0 -> v4.1:
// 1. ADDED ERA5 Surface Pressure (surface_pressure) as inverse barometer proxy.
// 2. ADDED Onshore Wind Vector Component (onshore_wind) engineered for Keta's 135deg SE shoreline orientation.
// 3. ADDED Rainfall Intensity Ratio (rain_intensity_ratio) to capture flashiness.
// 4. ADDED Tide Anomaly (tide_anomaly) to isolate sea-level surge from baseline tide.
// =====================================================================

// =====================================================================
// 1. REGION OF INTEREST (Keta Barrier & Lagoon Buffer)
// =====================================================================
var keta = ee.Geometry.Polygon([
  [[0.80, 5.75], [1.15, 5.75], [1.15, 6.10], [0.80, 6.10]]
]);

// Center map
Map.setCenter(0.975, 5.925, 11);
Map.setOptions("HYBRID");

// =====================================================================
// 2. DEM & DEM DERIVATIVES (SRTM 30m)
// =====================================================================
var srtm = ee.Image("NASA/NASADEM_HGT/001").select("elevation");
var dem  = srtm.clip(keta).rename("elevation");
var slope = ee.Terrain.slope(dem).rename("slope");
var aspect = ee.Terrain.aspect(dem).rename("aspect");

// JRC Global Surface Water (permanent water mask + dist_water)
var jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater");
var permWater = jrc.select("occurrence").gt(80).unmask(0);
var distWater = permWater.not().fastDistanceTransform(500).sqrt()
  .multiply(30).rename("dist_water");

// Binary elevation thresholds
var lt3 = dem.lt(3).rename("lt3");
var lt1 = dem.lt(1).rename("lt1");
var lt0 = dem.lt(0).rename("lt0");

// Terrain feature stack
var terrainStack = ee.Image.cat([dem, slope, aspect, distWater, lt3, lt1, lt0]);

// =====================================================================
// 3. EVENT CATALOGUE (19 Documented Events: 2019 - 2025)
// =====================================================================
var events = [
  { id:"jun2019", start:"2019-06-01", end:"2019-06-20", split:"train",      confidence:"MEDIUM" },
  { id:"jul2019", start:"2019-07-01", end:"2019-07-20", split:"train",      confidence:"MEDIUM" },
  { id:"jun2020", start:"2020-06-01", end:"2020-06-20", split:"train",      confidence:"MEDIUM" },
  { id:"jul2020", start:"2020-07-01", end:"2020-07-20", split:"train",      confidence:"MEDIUM" },
  { id:"may2021", start:"2021-05-01", end:"2021-05-15", split:"train",      confidence:"HIGH" },
  { id:"jun2021", start:"2021-06-01", end:"2021-06-15", split:"train",      confidence:"HIGH" },
  { id:"nov2021", start:"2021-11-10", end:"2021-11-20", split:"train",      confidence:"HIGH" },
  { id:"may2022", start:"2022-05-01", end:"2022-05-15", split:"train",      confidence:"MEDIUM" },
  { id:"jun2022", start:"2022-06-01", end:"2022-06-15", split:"train",      confidence:"MEDIUM" },
  { id:"jul2022", start:"2022-07-01", end:"2022-07-15", split:"train",      confidence:"MEDIUM" },
  { id:"jun2023", start:"2023-06-01", end:"2023-06-15", split:"test",       confidence:"HIGH" },
  { id:"jul2023", start:"2023-07-01", end:"2023-07-15", split:"test",       confidence:"HIGH" },
  { id:"sep2023", start:"2023-09-15", end:"2023-09-25", split:"test",       confidence:"HIGH" },
  { id:"oct2023", start:"2023-10-15", end:"2023-10-25", split:"test",       confidence:"HIGH" },
  { id:"feb2024", start:"2024-02-10", end:"2024-02-20", split:"validation", confidence:"HIGH" },
  { id:"may2024", start:"2024-05-15", end:"2024-05-25", split:"validation", confidence:"HIGH" },
  { id:"jan2025", start:"2025-01-10", end:"2025-01-20", split:"validation", confidence:"HIGH" },
  { id:"feb2025", start:"2025-02-05", end:"2025-02-15", split:"validation", confidence:"HIGH" },
  { id:"mar2025", start:"2025-03-01", end:"2025-03-10", split:"validation", confidence:"HIGH" },
  { id:"sep2025_market",     start:"2025-09-12", end:"2025-09-15", split:"validation", confidence:"MEDIUM" },
  { id:"jun2025_lawoshime",  start:"2025-06-25", end:"2025-07-05", split:"validation", confidence:"MEDIUM" },
  { id:"may2026_downpour",   start:"2026-05-06", end:"2026-05-18", split:"validation", confidence:"MEDIUM" },
  { id:"jun2026_floodgates", start:"2026-06-15", end:"2026-07-03", split:"validation", confidence:"MEDIUM" }
];

var trainIds      = ["jun2019","jul2019","jun2020","jul2020","may2021","jun2021","nov2021","may2022","jun2022","jul2022"];
var testIds       = ["jun2023","jul2023","sep2023","oct2023"];
var validationIds = ["feb2024","may2024","jan2025","feb2025","mar2025","sep2025_market","jun2025_lawoshime","may2026_downpour","jun2026_floodgates"];

// =====================================================================
// 3b. BASELINE LOOKBACK CORRECTION (Generalized Rule)
// =====================================================================
// For any event whose standard 12-day pre-event lookback falls within 14 days 
// of a prior catalogued flood's end date, the baseline lookback is extended 
// backward until it clears the prior event by at least 14 days.
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
// 4. TIDAL DICTIONARY (FES2022)
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
  "sep2025_market":{ max:0.682, range:1.300, spring:0 },
  "jun2025_lawoshime":{ max:0.658, range:1.493, spring:0 },
  "may2026_downpour":{ max:0.821, range:1.694, spring:1 },
  "jun2026_floodgates":{ max:0.736, range:1.621, spring:1 }
};

// =====================================================================
// 5. SENTINEL-1 SAR COLLECTION
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
// 6c. NEW CANDIDATE EVENTS -- SENTINEL-1 COVERAGE CHECK
// =====================================================================
var candidateEvents = [
  {id:"feb2025_agavedzi",     start:"2025-02-01", end:"2025-02-10"},
  {id:"sep2025_market",       start:"2025-09-12", end:"2025-09-15"},
  {id:"jun2025_lawoshime",    start:"2025-06-25", end:"2025-07-05"},
  {id:"jun2025_tidal_warning",start:"2025-06-27", end:"2025-06-30"},
  {id:"may2026_downpour",     start:"2026-05-06", end:"2026-05-10"},
  {id:"jun2026_floodgates",   start:"2026-06-15", end:"2026-07-03"}
];

print("=== NEW CANDIDATE EVENTS: SENTINEL-1 COVERAGE ===");
candidateEvents.forEach(function(e) {
  var t0 = ee.Date(e.start);
  var t1 = ee.Date(e.end);
  var beforeCount = s1.filterDate(t0.advance(-12, "day"), t0).size();
  var afterCount  = s1.filterDate(t0, t1).size();
  print(e.id, ee.Dictionary({before_images: beforeCount, after_images: afterCount}));
});

// =====================================================================
// 6. FLOOD LABEL -- OTSU THRESHOLDING
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
    var value   = aCount.multiply(bCount).multiply(aMean.subtract(bMean).pow(2));
    return ee.Number(ee.Algorithms.If(aCount.gt(0).and(bCount.gt(0)), value, 0));
  });

  var maxBssIdx = bss.indexOf(ee.List(bss).reduce(ee.Reducer.max()));
  return means.get([maxBssIdx]);
}

function getFloodLabel(event) {
  var t0 = ee.Date(event.get("start"));
  var t1 = ee.Date(event.get("end"));
  var bStart = ee.Date(event.get("base_start"));
  var bEnd   = ee.Date(event.get("base_end"));

  var beforeCol = s1.filterDate(bStart, bEnd);
  var duringCol = s1.filterDate(t0, t1);

  var dummyVH   = ee.Image.constant(0).rename("VH").selfMask();
  var beforeImg = beforeCol.merge(ee.ImageCollection([dummyVH])).median().unmask(0).rename("VH_before");
  var duringImg = duringCol.merge(ee.ImageCollection([dummyVH])).median().unmask(0).rename("VH_during");

  var diff = duringImg.subtract(beforeImg).rename("VH_diff");

  var hist = diff.reduceRegion({
    reducer: ee.Reducer.histogram({maxBuckets: 255}),
    geometry: keta,
    scale: 30,
    maxPixels: 1e8
  }).get("VH_diff");

  var otsuThresh = ee.Number(ee.Algorithms.If(
    hist,
    otsuThreshold(hist),
    -2.0
  ));

  var floodMask = diff.lt(otsuThresh)
    .and(permWater.not())
    .rename("flood")
    .toFloat();

  return floodMask.addBands(diff).set({
    "event_id": event.get("id"),
    "otsu_threshold": otsuThresh
  });
}

// =====================================================================
// 7. GPM IMERG V7
// =====================================================================
var imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation");

function getIMERG(t0, t1) {
  var dummy = ee.Image.constant(0).rename("precipitation").selfMask();
  var windowStart = t0.advance(-1, "day");
  var nDays = t1.difference(windowStart, "day").ceil().max(1);
  var dayOffsets = ee.List.sequence(0, nDays.subtract(1));

  var dailySums = ee.ImageCollection.fromImages(
    dayOffsets.map(function(d) {
      var dayStart = windowStart.advance(d, "day");
      var dayEnd   = dayStart.advance(1, "day");
      return imerg.filterDate(dayStart, dayEnd)
        .merge(ee.ImageCollection([dummy]))
        .sum().multiply(0.5).rename("precip_daily");
    })
  );

  var imerg24h = dailySums.max().unmask(0).rename("imerg_24h").toFloat();

  var imergCol = imerg.filterDate(windowStart, t1);
  var imergPeakIntensity = imergCol.merge(ee.ImageCollection([dummy]))
    .max().unmask(0).rename("imerg_peak_intensity").toFloat();

  var imerg3d = imergCol.merge(ee.ImageCollection([dummy]))
    .sum().multiply(0.5).unmask(0).rename("imerg_3d").toFloat();

  return ee.Image.cat([imerg24h, imergPeakIntensity, imerg3d]);
}

// =====================================================================
// 8. CHIRPS PRECIPITATION
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
// 9. ERA5 HOURLY -- ENHANCED WITH PRESSURE & ONSHORE WIND (v4.1)
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
  
  // v4.1 ADDITION: Onshore Wind Component perpendicular to Keta coastline (135deg SE)
  var onshoreWind = u.multiply(-0.7071).add(v.multiply(0.7071)).rename("onshore_wind");

  return ee.Image.cat([solar, u, v, windSpeed, windDir, onshoreWind]);
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
// 11. HYDRODYNAMIC BANDS (FES2022)
// =====================================================================
function getHydroBands(eventId) {
  var fes = ee.Dictionary(ee.Dictionary(FES2022_DATA).get(eventId));
  var maxTide = ee.Number(fes.get("max"));
  
  var tideMax   = ee.Image.constant(maxTide).rename("tide_max_fes");
  var tideRange = ee.Image.constant(ee.Number(fes.get("range"))).rename("tide_range_fes");
  var spring    = ee.Image.constant(ee.Number(fes.get("spring"))).rename("spring_flag_fes");
  
  // v4.1 ADDITION: Tide Anomaly (deviation from median baseline 0.75m)
  var tideAnomaly = ee.Image.constant(maxTide.subtract(0.75)).rename("tide_anomaly");

  return ee.Image.cat([tideMax, tideRange, spring, tideAnomaly]);
}

function getFreeboard(hydro, demImg) {
  return hydro.select("tide_max_fes").subtract(demImg).rename("freeboard_fes");
}

// =====================================================================
// 12. COMPOUND RISK BAND
// =====================================================================
function getCompoundRisk(imergBands, sm, hydro) {
  var normRain = imergBands.select("imerg_24h").divide(50).clamp(0, 1);
  var normSM   = sm.subtract(0.1).divide(0.4).clamp(0, 1);
  var normTide = hydro.select("tide_range_fes").subtract(1.258).divide(1.793 - 1.258).clamp(0, 1);
  return normRain.multiply(normSM).multiply(normTide).rename("compound_risk").toFloat();
}

// =====================================================================
// 13. SEASON FLAG
// =====================================================================
function getSeasonFlag(t) {
  var month = ee.Number(t.get("month")).toInt();
  var wet   = month.gte(4).and(month.lte(6)).or(month.gte(9).and(month.lte(11)));
  return ee.Image.constant(ee.Number(wet).toInt()).rename("season_wet").toFloat();
}

// =====================================================================
// 14. DRIVER LABEL CREATION (3-Class scheme: 0=no-flood, 1=rain, 2=coastal)
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

function normB(img, bandName, p10, p90) {
  var b = img.select(bandName);
  var span = ee.Number(p90).subtract(p10);
  var safeSpan = ee.Number(ee.Algorithms.If(span.gt(0.001), span, 1.0));
  return b.subtract(p10).divide(safeSpan).clamp(0, 1);
}

function getRFI(imergBands, rain, sm) {
  var normI = normB(imergBands, "imerg_24h",   rfciiPct.imerg_p10,  rfciiPct.imerg_p90);
  var normC = normB(rain,       "chirps_30d",  rfciiPct.chirps_p10, rfciiPct.chirps_p90);
  var normS = normB(sm,         "soil_moisture",rfciiPct.sm_p10,     rfciiPct.sm_p90);
  return normI.multiply(0.5).add(normC.multiply(0.3)).add(normS.multiply(0.2)).rename("RFI");
}

function getCII(hydro, demImg) {
  var freeboard = getFreeboard(hydro, demImg);
  var normFb = normB(freeboard, "freeboard_fes", rfciiPct.fb_p10, rfciiPct.fb_p90);
  var normTr = normB(hydro,     "tide_range_fes",rfciiPct.tr_p10, rfciiPct.tr_p90);
  return normFb.multiply(0.6).add(normTr.multiply(0.4)).rename("CII");
}

function getDriverLabel(floodMask, imergBands, rain, sm, hydro, demImg) {
  var rfi = getRFI(imergBands, rain, sm);
  var cii = getCII(hydro, demImg);

  var isRain    = rfi.gte(cii);
  var isCoastal = cii.gt(rfi);

  var driverCode = ee.Image.constant(0)
    .where(floodMask.eq(1).and(isRain), 1)
    .where(floodMask.eq(1).and(isCoastal), 2)
    .rename("flood_driver")
    .toInt();

  return driverCode.addBands(rfi).addBands(cii);
}

// =====================================================================
// 15. COMPOSITE FEATURE STACK BUILDER (v4.1 with 27 features)
// =====================================================================
function buildEventStack(event) {
  var t0 = ee.Date(event.start);
  var t1 = ee.Date(event.end);

  var floodImg   = getFloodLabel(ee.Dictionary(event));
  var imergBands = getIMERG(t0, t1);
  var rain       = getRain(t0);
  var era5Bands  = getERA5(t0);
  var sm         = getSM(t0);
  var hydro      = getHydroBands(event.id);
  var compound   = getCompoundRisk(imergBands, sm, hydro);
  var season     = getSeasonFlag(t0);
  var freeboard  = getFreeboard(hydro, dem);
  
  // v4.1 ADDITION: Rain Intensity Ratio
  var rainRatio  = imergBands.select("imerg_peak_intensity")
    .divide(rain.select("chirps_30d").add(1.0)).rename("rain_intensity_ratio");

  var driverImg  = getDriverLabel(
    floodImg.select("flood"), imergBands, rain, sm, hydro, dem
  );

  var stack = ee.Image.cat([
    terrainStack,        // 7 bands: elevation, slope, aspect, dist_water, lt3, lt1, lt0
    imergBands,          // 3 bands: imerg_24h, imerg_peak_intensity, imerg_3d
    rain,                // 2 bands: chirps_7d, chirps_30d
    era5Bands,           // 6 bands: solar_rad, u_wind_10m, v_wind_10m, wind_speed, wind_dir, onshore_wind
    sm,                  // 1 band:  soil_moisture
    hydro,               // 4 bands: tide_max_fes, tide_range_fes, spring_flag_fes, tide_anomaly
    freeboard,           // 1 band:  freeboard_fes
    compound,            // 1 band:  compound_risk
    season,              // 1 band:  season_wet
    rainRatio,           // 1 band:  rain_intensity_ratio
    floodImg.select("flood"),
    driverImg.select(["flood_driver", "RFI", "CII"])
  ]).clip(keta);

  return stack.set({
    "event_id": event.id,
    "split": event.split,
    "confidence": event.confidence
  });
}

// Build feature stacks for all 19 events
var eventStacks = events.map(function(e) {
  return buildEventStack(e);
});

// Export tasks setup
var allSamples = ee.FeatureCollection(eventStacks.map(function(stack) {
  var evId = stack.get("event_id");
  var split = stack.get("split");
  return stack.sample({
    region: keta,
    scale: 30,
    numPixels: 650,
    seed: 42,
    geometries: false
  }).map(function(f) {
    return f.set("event_id", evId).set("split", split);
  });
})).flatten();

Export.table.toDrive({
  collection: allSamples,
  description: "keta_samples_v4_1_all_splits",
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: allSamples.filter(ee.Filter.neq("flood_driver", 0)),
  description: "keta_samples_driver_v4_1_all_splits",
  fileFormat: "CSV"
});

print("=== GEE SCRIPT v4.1 LOADED SUCCESSFULLY ===");
