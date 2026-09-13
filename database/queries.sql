-- ============================================================================
-- queries.sql
--
-- Exploratory SQL analysis on the Ontario EWRB building energy dataset,
-- run directly against the buildings / energy_performance tables before
-- any Python-side modeling. Each query answers a specific question and is
-- commented with the business/data question it addresses.
--
-- Covers: JOIN, GROUP BY, aggregations, NULL handling, CASE, ORDER BY /
-- LIMIT, a CTE, and a window function.
-- ============================================================================


-- Q1: Which property types run the hottest on average energy intensity?
-- (JOIN + GROUP BY + aggregation + ORDER BY)
SELECT b.primary_property_type,
       ROUND(AVG(e.site_eui_gj_m2)::numeric, 2) AS avg_site_eui,
       COUNT(*) AS building_count
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.primary_property_type
ORDER BY avg_site_eui DESC;


-- Q2: Which individual buildings have the highest carbon intensity?
-- Useful as a starting point for the outlier investigation done later in
-- Python (see src/models/investigate_outliers.py).
-- (JOIN + ORDER BY + LIMIT)
SELECT b.ewrb_id, b.city, b.primary_property_type, e.ghg_intensity_kgco2e_m2
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
ORDER BY e.ghg_intensity_kgco2e_m2 DESC
LIMIT 10;


-- Q3: Which cities have the highest average water intensity?
-- (JOIN + GROUP BY + aggregation)
SELECT b.city,
       ROUND(AVG(e.water_intensity_m3_m2)::numeric, 2) AS avg_water_intensity
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.city
ORDER BY avg_water_intensity DESC;


-- Q4: How is the dataset distributed across property types?
-- (GROUP BY, no JOIN needed)
SELECT primary_property_type, COUNT(*) AS num_buildings
FROM buildings
GROUP BY primary_property_type
ORDER BY num_buildings DESC;


-- Q5: Which property types have the best Energy Star performance, among
-- buildings that were actually scored?
-- (JOIN + WHERE NULL handling + GROUP BY + aggregation)
SELECT b.primary_property_type,
       ROUND(AVG(e.energy_star_score)::numeric, 1) AS avg_star_score,
       COUNT(e.energy_star_score) AS scored_buildings
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.energy_star_score IS NOT NULL
GROUP BY b.primary_property_type
ORDER BY avg_star_score DESC;


-- Q6: How much does weather normalization change a building's reported
-- energy use? Large gaps indicate a building whose raw EUI is heavily
-- influenced by an unusually hot/cold year rather than its underlying
-- efficiency.
-- (JOIN + computed column + NULL handling + ORDER BY)
SELECT b.ewrb_id, b.city,
       e.site_eui_gj_m2,
       e.weather_normalized_site_eui_gj_m2,
       ROUND((e.site_eui_gj_m2 - e.weather_normalized_site_eui_gj_m2)::numeric, 2) AS diff
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.site_eui_gj_m2 IS NOT NULL
  AND e.weather_normalized_site_eui_gj_m2 IS NOT NULL
ORDER BY diff DESC;


-- Q7: Which building-years are missing key intensity readings?
-- Directly informs which rows would need to be excluded or imputed
-- before modeling.
-- (JOIN + OR'd NULL checks)
SELECT b.ewrb_id, b.city, b.primary_property_type,
       e.reporting_year, e.gas_intensity_gj_m2, e.water_intensity_m3_m2
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.gas_intensity_gj_m2 IS NULL
   OR e.water_intensity_m3_m2 IS NULL;


-- Q8: Classify every scored building into a simple efficiency tier.
-- (JOIN + CASE + NULLS LAST ordering)
SELECT b.ewrb_id, b.primary_property_type, e.energy_star_score,
    CASE
        WHEN e.energy_star_score >= 75 THEN 'High Performer'
        WHEN e.energy_star_score >= 50 THEN 'Average'
        WHEN e.energy_star_score IS NULL THEN 'Unscored'
        ELSE 'Low Performer'
    END AS efficiency_tier
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
ORDER BY e.energy_star_score DESC NULLS LAST;


-- Q9: Which cities perform better than the dataset-wide average GHG
-- intensity? Answers a two-step question (per-city average, then compare
-- to the overall average) that's awkward without a CTE.
-- (CTE + JOIN + GROUP BY + cross join against a single-row aggregate)
WITH city_avg AS (
    SELECT b.city,
           AVG(e.ghg_intensity_kgco2e_m2) AS avg_ghg
    FROM buildings b
    JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
    GROUP BY b.city
),
overall_avg AS (
    SELECT AVG(ghg_intensity_kgco2e_m2) AS overall_avg_ghg
    FROM energy_performance
)
SELECT c.city, ROUND(c.avg_ghg::numeric, 2) AS city_avg_ghg
FROM city_avg c, overall_avg o
WHERE c.avg_ghg > o.overall_avg_ghg
ORDER BY city_avg_ghg DESC;


-- Q10: Rank every building's Site EUI against others of the same property
-- type. Useful for spotting a building that's an outlier relative to its
-- peers, not just relative to the whole dataset.
-- (JOIN + window function: RANK() OVER PARTITION BY)
SELECT b.ewrb_id, b.city, b.primary_property_type,
       e.site_eui_gj_m2,
       RANK() OVER (
           PARTITION BY b.primary_property_type
           ORDER BY e.site_eui_gj_m2 DESC
       ) AS eui_rank_in_type
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.site_eui_gj_m2 IS NOT NULL
ORDER BY b.primary_property_type, eui_rank_in_type;