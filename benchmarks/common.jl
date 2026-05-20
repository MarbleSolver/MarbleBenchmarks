using Dates

function save_df_csv(df, results_dir, name)
    mkpath(results_dir)
    dt = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    CSV.write(joinpath(results_dir, "$(name)_$dt.csv"), df)
end