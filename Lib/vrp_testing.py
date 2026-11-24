import os
import folium
import openrouteservice as ors

# restaurants (lat, lon)
coordinates = [
    [51.481329, -0.010813],
    [51.481042, -0.008780],
    [51.479680, -0.0091387]
]

# map center (lat, lon)key

central_location = [51.481474, -0.009106]

# FOLIUM MAP
m = folium.Map(location=central_location, tiles="cartodbpositron", zoom_start=14)

for coord in coordinates:
    folium.Marker(location=coord).add_to(m)

m.save("map.html")

#-----------------------------------------------------------------------------------------------------------------#


# ORS client
client = ors.Client(key=os.environ["ORS_API_KEY"])  # set ORS_API_KEY env var

# --------------------------------------------------------
# Coordinates must be converted properly:
# ORS → [lon, lat]
# Folium → [lat, lon]

def to_ors(latlon):
    return [latlon[1], latlon[0]]  # convert (lat,lon) → (lon,lat)

# Starting positions (lat, lon)
starting_position1_latlon = [51.477890, -0.012200]
starting_position2_latlon = [51.482155, -0.005159]

# Convert for ORS
starting_position1 = to_ors(starting_position1_latlon)
starting_position2 = to_ors(starting_position2_latlon)

# NEW MAP
m1 = folium.Map(location=central_location, tiles="cartodbpositron", zoom_start=14)

# Add restaurant markers
for coord in coordinates:
    folium.Marker(location=coord).add_to(m1)

# Add bike starting marker
folium.Marker(location=starting_position1_latlon, icon=folium.Icon(color="red")).add_to(m1)

# HOME / END LOCATION
home1_latlon = [51.481091, -0.021006]
home1 = to_ors(home1_latlon)

folium.Marker(location=home1_latlon, icon=folium.Icon(color="green")).add_to(m1)

# VEHICLES
vehicles = [
    ors.optimization.Vehicle(
        id=1,
        profile='driving-car',
        start=starting_position1,
        end=home1,
        capacity=[5]
    ),
    ors.optimization.Vehicle(
        id=2,
        profile='driving-car',
        start=starting_position2,
        end=home1,
        capacity=[5]
    )
]


# JOBS (convert each restaurant to ORS format)
jobs = [
    ors.optimization.Job(
        id=i,
        location=to_ors(coord),
        amount=[1]
    )
    for i, coord in enumerate(coordinates)
]

# OPTIMIZE
optimized = client.optimization(jobs=jobs, vehicles=vehicles, geometry=True)

# ROUTES ON MAP
for route in optimized['routes']:
    decoded = ors.convert.decode_polyline(route['geometry'])['coordinates']
    polyline = [(lat, lon) for lon, lat in decoded]  # convert back to folium format
    folium.PolyLine(polyline, color="blue", weight=4).add_to(m1)
    print(f"Vehicle {route['vehicle']} total time: {route['duration'] / 60:.2f} minutes")

m1.save("m1.html")
