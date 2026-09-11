SELECT b.primary_property_type,
       ROUND(AVG(e.site_eui_gj_m2)::numeric, 2) AS avg_site_eui,
       COUNT(*) AS building_count
FROM buildings b
JOIN energy_performance e ON b.ewrb_id = e.ewrb_id
GROUP BY b.primary_property_type
ORDER BY avg_site_eui DESC;