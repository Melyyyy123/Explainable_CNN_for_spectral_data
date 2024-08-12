f_list <- ls(pattern = "dps", envir = .GlobalEnv)

for (name in f_list) {
  # Get the dataframe
  df <- get(name, envir = .GlobalEnv)
  
  # Create the filename by appending ".csv" to the dataframe name
  filename <- paste0("csv/", name, ".csv")
  
  # Save the dataframe to a CSV file
  write.csv(df, file = filename, row.names = FALSE)
  
  cat(sprintf("Dataframe '%s' saved to '%s'\n", name, filename))
}
