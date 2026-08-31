library(httr)
library(jsonlite)

#Busqueda para NVIDIA el 1 de junio 2026
url <- "https://api.gdeltproject.org/api/v2/doc/doc?query=NVIDIA&mode=ArtList&format=json&maxrecords=250&startdatetime=20260601000000&enddatetime=20260601235959"

#Buscamos la información y arreglamos encoding
txt <- GET(url) |>
  content("text", encoding = "UTF-8")

#Pasamos desde json
res <- fromJSON(txt, flatten = TRUE)

#Convertimos en data.frame
articulos <- res$articles