#%%
import folium

print(f'folium {folium.__version__}')
# %% 원광대 지도에 찍어보기
wku_location = (35.968726,126.957917) # 원대 프라임관 위도, 경도
# %%
m = folium.Map(location=wku_location,zoom_start = 16)
# %%
m
#m.show_in_browser < 무한로딩 되니까 ctrl+c로 나가기
# %%
folium.Marker(
    location = wku_location,
    tooltip = "click me",
    popup = "프라임관",
    icon = folium.Icon("cloud")
).add_to(m)
m
# %%
