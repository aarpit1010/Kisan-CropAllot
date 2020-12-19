from train import train_model
from predict import allot_crops

print("Enter district name, season and area available")
dist_name = str(input())
season = str(input())
area = float(input())
dist_name = dist_name.upper()
season = "{:<11}".format(season)
train_model(dist_name)
crops_alloted = allot_crops(dist_name, season, area)
print(crops_alloted)
