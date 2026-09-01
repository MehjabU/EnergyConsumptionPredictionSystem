CREATE TABLE buildings(
	ewrb_id VARCHAR(10) PRIMARY KEY,
	city VARCHAR(30),
	postal_code VARCHAR(10),
	primary_property_type VARCHAR(50),
	self_property_type VARCHAR(50),
	largest_property_type VARCHAR(50),
	all_property_types TEXT,
	third_party_certification VARCHAR(50)
);

CREATE TABLE energy_performance( 
	performance_id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, 
	ewrb_id VARCHAR(10) REFERENCES buildings(ewrb_id), 
	reporting_year INT, 
	electricity_intensity_gj_m2 DOUBLE PRECISION, 
	gas_intensity_gj_m2 DOUBLE PRECISION, 
	water_intensity_m3_m2 DOUBLE PRECISION, 
	indoor_water_intensity_m3_m2 DOUBLE PRECISION, 
	site_eui_gj_m2 DOUBLE PRECISION, 
	weather_normalized_site_eui_gj_m2 DOUBLE PRECISION, 
	source_eui_gj_m2 DOUBLE PRECISION, 
	weather_normalized_source_eui_gj_m2 DOUBLE PRECISION, 
	ghg_intensity_kgco2e_m2 DOUBLE PRECISION, 
	energy_star_score INT 
);
SELECT * FROM buildings;
SELECT * FROM energy_performance

ALTER TABLE energy_performance
ADD CONSTRAINT energy_performance_ewrb_year_unique
UNIQUE (ewrb_id, reporting_year);