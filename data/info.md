The CSV files in this directory are ignored by git, but should contain the files that we're working on:

- `bastin_db_cleaned.csv` - full dataset without the rows with missing lon/lat values. Row headers `location_x` and `location_y` have been replaced with `longitude` and `latitude` for compatibility with Google Earth Engine asset import.
- `by_region/...` - the dataset splitted into regions, files are named `region_{}.csv`.

`xsv` was used to split up by region:

```bash
cd dataxsv partition --filename region_{}.csv dryland_assessment_region by_region bastin_db_cleaned.csv
cd by_region
# did that, but they are not used currently...
ls | xargs -I data /bin/zsh -c 'xsv split -s 5300 --filename "data_{}.csv" splits data'
```
