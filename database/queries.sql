-- Average site EUI by property type
SELECT b.primary_property_type,
       ROUND(AVG(e.site_eui_gj_m2)::numeric, 2) AS avg_site_eui,
       COUNT(*) AS building_count
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.primary_property_type
ORDER BY avg_site_eui DESC;

-- identify buildings with the highest GHG intensity
SELECT b.ewrb_id, b.city, b.primary_property_type, e.ghg_intensity_kgco2e_m2
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
ORDER BY e.ghg_intensity_kgco2e_m2 DESC
LIMIT 10;

-- avg water intensity by city
SELECT b.city,
       ROUND(AVG(e.water_intensity_m3_m2)::numeric, 2) AS avg_water_intensity
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.city
ORDER BY avg_water_intensity DESC;

-- counting buildings by primary property type
SELECT primary_property_type, COUNT(*) AS num_buildings
FROM buildings
GROUP BY primary_property_type
ORDER BY num_buildings DESC;

-- average energy star score by property type
SELECT b.primary_property_type,
       ROUND(AVG(e.energy_star_score)::numeric, 1) AS avg_star_score,
       COUNT(e.energy_star_score) AS scored_buildings
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.energy_star_score IS NOT NULL
GROUP BY b.primary_property_type
ORDER BY avg_star_score DESC;

-- Comparing site EUI and weather-normalized site EUI
SELECT b.ewrb_id, b.city,
       e.site_eui_gj_m2,
       e.weather_normalized_site_eui_gj_m2,
       ROUND((e.site_eui_gj_m2 - e.weather_normalized_site_eui_gj_m2)::numeric, 2) AS diff
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.site_eui_gj_m2 IS NOT NULL
  AND e.weather_normalized_site_eui_gj_m2 IS NOT NULL
ORDER BY diff DESC;

-- identifying rows with missing gas or water intensity values
SELECT b.ewrb_id, b.city, b.primary_property_type,
       e.reporting_year, e.gas_intensity_gj_m2, e.water_intensity_m3_m2
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
WHERE e.gas_intensity_gj_m2 IS NULL
   OR e.water_intensity_m3_m2 IS NULL;

-- classifying buildings based on energy star score (tiers)
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