SELECT b.primary_property_type,
       ROUND(AVG(e.site_eui_gj_m2)::numeric, 2) AS avg_site_eui,
       COUNT(*) AS building_count
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.primary_property_type
ORDER BY avg_site_eui DESC;

SELECT b.ewrb_id, b.city, b.primary_property_type, e.ghg_intensity_kgco2e_m2
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
ORDER BY e.ghg_intensity_kgco2e_m2 DESC
LIMIT 10;

SELECT b.city,
       ROUND(AVG(e.water_intensity_m3_m2)::numeric, 2) AS avg_water_intensity
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.city
ORDER BY avg_water_intensity DESC;