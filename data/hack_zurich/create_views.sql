CREATE VIEW accessability_bus AS
    SELECT v.value as "access_by_bus", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Access by bus [% of inhabitants]';

CREATE VIEW accessability_train AS
    SELECT v.value as "access_by_suburban_train", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Access by suburban train [% of inhabitants]';

CREATE VIEW accessability_train_and_bus AS
    SELECT v.value as "access_by_suburban_train_and_bus", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Accessibility by suburban train+bus [% of inhabitants]';

CREATE VIEW number_of_passenger_cars AS
    SELECT v.value as "passenger_cars_per_1000_inhabitants", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Passenger cars per 1000 inhabitants [no.]';

CREATE VIEW distance_next_stop AS
    SELECT v.value as "distance_to_next_stop", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Distance to the next stop [m]';

CREATE VIEW public_transport_share AS
    SELECT v.value as "public_transport_share_modal_split", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'public transport share (modal split) [%]';

CREATE VIEW miv_share AS
    SELECT v.value as "miv_share_modal_split", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'MIV share (modal split) [%]';

CREATE VIEW amount_new_pw_registrations AS
    SELECT v.value as "amount_new_pw_registrations_per_1000_inhabitants", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'PW new registrations per 1000 inhabitants [amount]';

CREATE VIEW share_hybrid_cars AS
    SELECT v.value as "share_of_hybrid_cars", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Hybrid motor cars stock [%]';

CREATE VIEW share_electric_cars AS
    SELECT v.value as "share_electric_cars", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'Electric motor cars stock [%]';

CREATE VIEW new_hybrid_car_registrations AS
    SELECT v.value as "new_hybrid_car_registrations", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'New registrations of hybrid motor cars [%]';

CREATE VIEW new_electric_car_registrations AS
    SELECT v.value as "new_electric_car_registrations", v.year, v.spatialunit_id
    FROM indicators i inner join indicator_values2 v on i.indicator_id = v.indicator_id
    WHERE trim(i.name) = 'New registrations electric motor cars [%]';
