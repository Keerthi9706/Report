# Dashboard
# Setup: shiny::runApp()   

library(shiny)
library(shinydashboard)
library(DT)
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(randomForest)


# Load data and train workflow once at app startup
set.seed(123)
message("Training pipeline — please wait...")

csv_path <- "Crop_Recommendation.csv"
if (!file.exists(csv_path)) {
  stop("Crop_Recommendation.csv was not found in the app folder.")
}

df <- read.csv(csv_path, stringsAsFactors = FALSE)

features <- c(
  "Nitrogen", "Phosphorus", "Potassium",
  "Temperature", "Humidity", "pH_Value", "Rainfall"
)

required_cols <- c(features, "Crop")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(
    paste(
      "The dataset is missing required columns:",
      paste(missing_cols, collapse = ", ")
    )
  )
}

# 1. Standardize predictors
preproc <- preProcess(df[features], method = c("center", "scale"))
dfZ <- predict(preproc, df[features])

# 2. K-means clustering with k = 4
km <- kmeans(dfZ, centers = 4, nstart = 25)
df$Cluster <- factor(km$cluster)

# 3. 80/20 stratified split
idx <- createDataPartition(df$Cluster, p = 0.8, list = FALSE)
train_data <- df[idx, ]
test_data  <- df[-idx, ]

# 4. Random Forest with 5-fold CV tuning
rf_fit <- caret::train(
  x = train_data[features],
  y = train_data$Cluster,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(mtry = 2:6),
  importance = TRUE
)

# 5. Test-set evaluation
rf_pred <- predict(rf_fit, test_data[features])
rf_cm <- confusionMatrix(data = rf_pred, reference = test_data$Cluster)

# 6. Cluster -> top crops summary
cluster_crops <- df %>%
  group_by(Cluster, Crop) %>%
  tally() %>%
  arrange(Cluster, desc(n))

# 7. PCA view for clustering tab
pca <- prcomp(dfZ, center = FALSE, scale. = FALSE)
pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  Cluster = df$Cluster
)

message("Training complete. Test accuracy: ",
        round(as.numeric(rf_cm$overall["Accuracy"]), 4))

# Colors
pal_cluster <- c(
  "1" = "#2C5F2D",
  "2" = "#97BC62",
  "3" = "#B85042",
  "4" = "#065A82"
)

# Report metrics for comparison panel
metrics_df <- data.frame(
  Model = rep(c("Decision Tree", "Random Forest", "k-NN"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1"), 3),
  Value = c(
    0.9728, 0.9724, 0.9776, 0.9748,
    0.9819, 0.9842, 0.9856, 0.9847,
    0.9819, 0.9847, 0.9843, 0.9844
  )
)

# UI
ui <- dashboardPage(
  skin = "green",
  dashboardHeader(title = "Crop Group Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("seedling")),
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar")),
      menuItem("Clustering", tabName = "cluster", icon = icon("circle-nodes")),
      menuItem("Model Comparison", tabName = "models", icon = icon("trophy")),
      menuItem("Predict", tabName = "predict", icon = icon("magnifying-glass"))
    )
  ),
  dashboardBody(
    tags$head(tags$style(HTML("
      .small-box.bg-olive { background-color:#2C5F2D !important; color:#fff; }
      .box.box-solid.box-primary>.box-header { background:#2C5F2D; }
      .box.box-solid.box-primary { border:1px solid #2C5F2D; }
    "))),
    tabItems(
      # ---- Overview ----
      tabItem(
        "overview",
        fluidRow(
          valueBox(nrow(df), "Observations", icon = icon("database"), color = "olive"),
          valueBox(length(features), "Numeric predictors", icon = icon("vials"), color = "green"),
          valueBox(nlevels(df$Cluster), "Agronomic clusters", icon = icon("layer-group"), color = "teal")
        ),
        fluidRow(
          box(
            title = "About this dashboard",
            status = "primary", solidHeader = TRUE, width = 12,
            p("This dashboard mirrors the project workflow:"),
            tags$ol(
              tags$li("K-means clusters the standardized soil and climate variables into four agronomic groups."),
              tags$li("A Random Forest classifier predicts cluster membership from the original predictors.")
            ),
            p("Use the sidebar to explore the data, view the cluster structure, compare models, and test a new agronomic profile."),
            p(tags$em("This app predicts agronomic clusters, not exact crop names."))
          )
        )
      ),

      # EDA
      tabItem(
        "eda",
        fluidRow(
          box(
            title = "Variable distribution", status = "primary", solidHeader = TRUE, width = 4,
            selectInput("eda_var", "Variable:", choices = features),
            checkboxInput("eda_split", "Color by cluster", value = TRUE)
          ),
          box(
            title = NULL, status = "primary", solidHeader = FALSE, width = 8,
            plotOutput("eda_hist", height = 320)
          )
        ),
        fluidRow(
          box(
            title = "Summary statistics", status = "primary", solidHeader = TRUE, width = 12,
            DTOutput("eda_summary")
          )
        ),
        fluidRow(
          box(
            title = "Correlation heatmap", status = "primary", solidHeader = TRUE, width = 12,
            plotOutput("eda_corr", height = 380)
          )
        )
      ),

      # Clustering
      tabItem(
        "cluster",
        fluidRow(
          box(
            title = "PCA view of the four clusters", status = "primary", solidHeader = TRUE, width = 8,
            plotOutput("pca_plot", height = 460)
          ),
          box(
            title = "Cluster sizes", status = "primary", solidHeader = TRUE, width = 4,
            tableOutput("cluster_sizes")
          )
        ),
        fluidRow(
          box(
            title = "Most represented crops in each cluster", status = "primary", solidHeader = TRUE, width = 12,
            DTOutput("cluster_crops_tbl")
          )
        )
      ),

      # Model Comparison
      tabItem(
        "models",
        fluidRow(
          box(
            title = "Test-set performance", status = "primary", solidHeader = TRUE, width = 7,
            plotOutput("model_bars", height = 360),
            tags$small("Decision Tree and k-NN metrics are shown from the project report; Random Forest values match the selected final model.")
          ),
          box(
            title = "Selected final model", status = "primary", solidHeader = TRUE, width = 5,
            h2("Random Forest", style = "color:#2C5F2D;margin-top:0;"),
            h3(textOutput("rf_acc", inline = TRUE)), p("test accuracy"),
            h3("0.9847"), p("macro F1 (project result)"),
            p(tags$em("Chosen for the best overall balance of recall, macro F1, and interpretability."))
          )
        ),
        fluidRow(
          box(
            title = "Random Forest confusion matrix (test set)", status = "primary",
            solidHeader = TRUE, width = 8, plotOutput("conf_plot", height = 360)
          ),
          box(
            title = "Feature importance", status = "primary", solidHeader = TRUE, width = 4,
            plotOutput("imp_plot", height = 360)
          )
        )
      ),

      # Predict
      tabItem(
        "predict",
        fluidRow(
          box(
            title = "Enter agronomic conditions", status = "primary", solidHeader = TRUE, width = 5,
            numericInput("Nitrogen", "Nitrogen (N)", value = 90, min = 0, max = 200),
            numericInput("Phosphorus", "Phosphorus (P)", value = 42, min = 0, max = 200),
            numericInput("Potassium", "Potassium (K)", value = 43, min = 0, max = 250),
            numericInput("Temperature", "Temperature (\u00B0C)", value = 21, min = 5, max = 50),
            numericInput("Humidity", "Humidity (%)", value = 82, min = 0, max = 100),
            numericInput("pH_Value", "Soil pH", value = 6.5, min = 3, max = 10, step = 0.1),
            numericInput("Rainfall", "Rainfall (mm)", value = 200, min = 0, max = 350),
            actionButton(
              "go", "Predict cluster",
              icon = icon("play"),
              style = "background:#2C5F2D;color:#fff;border:none;"
            )
          ),
          box(
            title = "Predicted agronomic cluster", status = "primary", solidHeader = TRUE, width = 7,
            uiOutput("pred_box"),
            br(),
            h4("Class probabilities"),
            plotOutput("prob_plot", height = 220),
            br(),
            h4("Crops typical of this cluster"),
            tableOutput("pred_crops")
          )
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {

  # ---- EDA ----
  output$eda_hist <- renderPlot({
    g <- ggplot(df, aes(x = .data[[input$eda_var]]))
    if (input$eda_split) {
      g <- g +
        geom_histogram(
          aes(fill = Cluster), bins = 30,
          color = "white", alpha = 0.85
        ) +
        scale_fill_manual(values = pal_cluster)
    } else {
      g <- g +
        geom_histogram(bins = 30, fill = "#2C5F2D", color = "white")
    }

    g +
      theme_minimal(base_size = 13) +
      labs(x = input$eda_var, y = "Count")
  })

  output$eda_summary <- renderDT({
    summary_df <- df %>%
      dplyr::select(all_of(features)) %>%
      summarise(
        across(
          everything(),
          list(
            mean = ~ round(mean(.), 2),
            sd = ~ round(sd(.), 2),
            min = ~ round(min(.), 2),
            max = ~ round(max(.), 2)
          )
        )
      ) %>%
      pivot_longer(
        everything(),
        names_to = c("Variable", ".value"),
        names_sep = "_(?=[^_]+$)"
      )

    datatable(summary_df, options = list(dom = "t", paging = FALSE), rownames = FALSE)
  })

  output$eda_corr <- renderPlot({
    cm <- cor(df[features])
    cm_long <- as.data.frame(as.table(cm))

    ggplot(cm_long, aes(Var1, Var2, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = sprintf("%.2f", Freq)), size = 3.5) +
      scale_fill_gradient2(
        low = "#B85042", mid = "white", high = "#2C5F2D",
        midpoint = 0, limits = c(-1, 1), name = "r"
      ) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = NULL, y = NULL)
  })

  # Clustering 
  output$pca_plot <- renderPlot({
    ggplot(pca_df, aes(PC1, PC2, color = Cluster)) +
      geom_point(alpha = 0.75, size = 2) +
      scale_color_manual(values = pal_cluster) +
      theme_minimal(base_size = 13) +
      labs(title = "K-means clusters projected onto the first two principal components")
  })

  output$cluster_sizes <- renderTable({
    df %>% count(Cluster, name = "Count")
  })

  output$cluster_crops_tbl <- renderDT({
    top_crops_tbl <- cluster_crops %>%
      group_by(Cluster) %>%
      summarise(
        `Top crops (count)` = paste0(Crop, " (", n, ")", collapse = ", "),
        .groups = "drop"
      )

    datatable(top_crops_tbl, options = list(dom = "t", paging = FALSE), rownames = FALSE)
  })

  # Model comparison
  output$rf_acc <- renderText({
    sprintf("%.2f%%", as.numeric(rf_cm$overall["Accuracy"]) * 100)
  })

  output$model_bars <- renderPlot({
    ggplot(metrics_df, aes(Metric, Value, fill = Model)) +
      geom_col(position = "dodge", width = 0.7) +
      coord_cartesian(ylim = c(0.96, 1.00)) +
      scale_fill_manual(
        values = c(
          "Decision Tree" = "#B85042",
          "k-NN" = "#97BC62",
          "Random Forest" = "#2C5F2D"
        )
      ) +
      geom_text(
        aes(label = sprintf("%.4f", Value)),
        position = position_dodge(width = 0.7),
        vjust = -0.4, size = 3
      ) +
      theme_minimal(base_size = 13) +
      labs(x = NULL, y = NULL)
  })

  output$conf_plot <- renderPlot({
    cm_df <- as.data.frame(rf_cm$table)
    names(cm_df) <- c("Prediction", "Reference", "Freq")

    ggplot(cm_df, aes(Reference, Prediction, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), color = "#1A2E1A", size = 5, fontface = "bold") +
      scale_fill_gradient(low = "#F5F5F5", high = "#2C5F2D") +
      theme_minimal(base_size = 13) +
      labs(x = "Actual cluster", y = "Predicted cluster")
  })

  output$imp_plot <- renderPlot({
    imp <- varImp(rf_fit)$importance
    imp$Variable <- rownames(imp)
    imp$Overall <- rowMeans(imp[, setdiff(names(imp), "Variable"), drop = FALSE])

    ggplot(imp, aes(reorder(Variable, Overall), Overall)) +
      geom_col(fill = "#2C5F2D") +
      coord_flip() +
      theme_minimal(base_size = 12) +
      labs(x = NULL, y = "Importance")
  })

  # Predict
  prediction <- eventReactive(input$go, {
    new_data <- data.frame(
      Nitrogen = input$Nitrogen,
      Phosphorus = input$Phosphorus,
      Potassium = input$Potassium,
      Temperature = input$Temperature,
      Humidity = input$Humidity,
      pH_Value = input$pH_Value,
      Rainfall = input$Rainfall
    )

    list(
      class = predict(rf_fit, new_data),
      probs = predict(rf_fit, new_data, type = "prob")
    )
  })

  output$pred_box <- renderUI({
    req(prediction())
    cl <- as.character(prediction()$class)

    div(
      style = sprintf(
        "background:%s;color:#fff;padding:18px;border-radius:8px;text-align:center;",
        pal_cluster[cl]
      ),
      h2(paste("Cluster", cl), style = "margin:0;"),
      p("predicted from the agronomic inputs", style = "margin:0;")
    )
  })

  output$prob_plot <- renderPlot({
    req(prediction())
    pr <- prediction()$probs
    prob_df <- data.frame(Cluster = names(pr), Prob = as.numeric(pr[1, ]))

    ggplot(prob_df, aes(Cluster, Prob, fill = Cluster)) +
      geom_col(width = 0.6) +
      scale_fill_manual(values = pal_cluster) +
      geom_text(aes(label = sprintf("%.2f", Prob)), vjust = -0.3) +
      ylim(0, 1) +
      theme_minimal(base_size = 13) +
      theme(legend.position = "none") +
      labs(x = NULL, y = "Probability")
  })

  output$pred_crops <- renderTable({
    req(prediction())
    cl <- as.character(prediction()$class)

    cluster_crops %>%
      filter(Cluster == cl) %>%
      head(8) %>%
      rename(`Crop` = Crop, `Count in cluster` = n) %>%
      dplyr::select(Crop, `Count in cluster`)
  })
}

shinyApp(ui, server)