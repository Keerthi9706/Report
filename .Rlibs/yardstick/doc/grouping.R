## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
library(yardstick)
library(dplyr)

data("hpc_cv")

## -----------------------------------------------------------------------------
tibble(hpc_cv)

## -----------------------------------------------------------------------------
set.seed(1)

hpc <-
  tibble(hpc_cv) |>
  mutate(batch = sample(c("a", "b"), nrow(hpc_cv), replace = TRUE)) |>
  select(-c(VF, F, M, L))

hpc

## -----------------------------------------------------------------------------
hpc |> 
  filter(Resample == "Fold01") |>
  accuracy(obs, pred)

## -----------------------------------------------------------------------------
hpc |> 
  group_by(Resample) |>
  accuracy(obs, pred)

## -----------------------------------------------------------------------------
hpc |> 
  filter(Resample == "Fold01")

## -----------------------------------------------------------------------------
acc_by_group <- 
  hpc |> 
  filter(Resample == "Fold01") |>
  group_by(batch) |>
  accuracy(obs, pred)

acc_by_group

## -----------------------------------------------------------------------------
diff(c(acc_by_group$.estimate[2], acc_by_group$.estimate[1]))

## -----------------------------------------------------------------------------
accuracy_diff <-
  new_groupwise_metric(
    fn = accuracy,
    name = "accuracy_diff",
    aggregate = function(acc_by_group) {
      diff(c(acc_by_group$.estimate[2], acc_by_group$.estimate[1]))
    }
  )

## -----------------------------------------------------------------------------
class(accuracy_diff)

## -----------------------------------------------------------------------------
accuracy_diff_by_batch <- accuracy_diff(batch)

## -----------------------------------------------------------------------------
class(accuracy)

class(accuracy_diff_by_batch)

## -----------------------------------------------------------------------------
hpc |> 
  filter(Resample == "Fold01") |>
  accuracy_diff_by_batch(obs, pred)

## -----------------------------------------------------------------------------
acc_ms <- metric_set(accuracy, accuracy_diff_by_batch)

hpc |> 
  filter(Resample == "Fold01") |>
  acc_ms(truth = obs, estimate = pred)

## -----------------------------------------------------------------------------
hpc |> 
  group_by(Resample) |>
  accuracy_diff_by_batch(obs, pred)

