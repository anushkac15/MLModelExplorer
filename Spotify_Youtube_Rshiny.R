# Load necessary libraries
library(caret)
library(e1071)
library(class)
library(rpart)
library(rpart.plot)
library(naivebayes)
library(ggplot2)
library(gridExtra)
library(shiny)
library(pROC)

# Increase the maximum upload size (50MB)
options(shiny.maxRequestSize = 50 * 1024^2)  # Set max upload size to 50MB

# Define UI for the app
ui <- navbarPage(
  # Application title
  title = "Model Evaluation and Clustering Visualization",
  
  # File upload tab
  tabPanel("Upload Data",
           fluidPage(
             titlePanel("Upload CSV File for Model Evaluation"),
             sidebarLayout(
               sidebarPanel(
                 fileInput("datafile", "Choose CSV File", accept = ".csv"),
                 helpText("Upload a CSV file containing your dataset.")
               ),
               mainPanel(
                 h4("Data Preview"),
                 tableOutput("data_preview")
               )
             )
           )
  ),
  
  # Navigation items for different visualizations
  tabPanel("Confusion Matrix",
           h3("Confusion Matrix Plots"),
           plotOutput("conf_matrix_plot")
  ),
  
  tabPanel("Model Evaluation Metrics",
           h3("Model Evaluation Metrics"),
           tableOutput("metrics_table"),
           h3("Evaluation Metrics Plot"),
           plotOutput("metrics_plot")
  ),
  
  tabPanel("K-Means Clustering",
           h3("K-Means Clustering Visualization"),
           plotOutput("kmeans_plot")
  ),
  
  tabPanel("ROC Curve",
           h3("ROC Curve Plot"),
           plotOutput("roc_curve_plot")
  )
)
# Define server logic
server <- function(input, output) {
  
  # Reactive expression to load data when the user uploads a file
  data <- reactive({
    req(input$datafile)
    read.csv(input$datafile$datapath)
  })
  
  # Display a preview of the uploaded data
  output$data_preview <- renderTable({
    req(data())
    head(data())  # Show the first few rows of the uploaded dataset
  })
  
  # Reactive expression to preprocess data and train models when data is available
  results <- reactive({
    # Load and preprocess data
    data_df <- data()
    data_df$official_video <- as.factor(data_df$official_video)
    target_levels <- c("True", "False")
    data_df$official_video <- factor(data_df$official_video, levels = target_levels)
    
    audio_features <- c("Danceability", "Energy", "Loudness", "Speechiness", 
                        "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "official_video")
    df <- data_df[, audio_features]
    df <- na.omit(df)
    
    # Train-Test Split
    set.seed(123)
    trainIndex <- createDataPartition(df$official_video, p = 0.8, list = FALSE)
    trainData <- df[trainIndex, ]
    testData <- df[-trainIndex, ]
    
    # Define a function to calculate evaluation metrics
    evaluate_model <- function(pred, actual) {
      cm <- confusionMatrix(pred, actual)
      accuracy <- cm$overall['Accuracy']
      error_rate <- 1 - accuracy
      precision <- cm$byClass['Precision']
      recall <- cm$byClass['Sensitivity']
      f1_score <- cm$byClass['F1']
      metrics <- c(accuracy, error_rate, precision, recall, f1_score)
      names(metrics) <- c("Accuracy", "Error Rate", "Precision", "Recall", "F1 Score")
      return(metrics)
    }
    
    # Initialize metrics storage as a list
    metrics_list <- list()
    
    # Train models and evaluate them
    # 1. KNN Model
    trainData_scaled <- scale(trainData[, -10])
    testData_scaled <- scale(testData[, -10])
    k_value <- 5
    knn_pred <- knn(train = trainData_scaled, test = testData_scaled, cl = trainData$official_video, k = k_value)
    knn_pred <- factor(knn_pred, levels = target_levels)
    knn_metrics <- evaluate_model(knn_pred, testData$official_video)
    metrics_list[["KNN"]] <- knn_metrics
    
    # 2. Naive Bayes Model
    nb_model <- naive_bayes(official_video ~ ., data = trainData)
    nb_pred <- predict(nb_model, testData)
    nb_pred <- factor(nb_pred, levels = target_levels)
    nb_metrics <- evaluate_model(nb_pred, testData$official_video)
    metrics_list[["Naive Bayes"]] <- nb_metrics
    
    # 3. Decision Tree Model
    dt_model <- rpart(official_video ~ ., data = trainData, method = "class")
    dt_pred <- predict(dt_model, testData, type = "class")
    dt_pred <- factor(dt_pred, levels = target_levels)
    dt_metrics <- evaluate_model(dt_pred, testData$official_video)
    metrics_list[["Decision Tree"]] <- dt_metrics
    
    # 4. Support Vector Machine (SVM)
    svm_model <- svm(official_video ~ ., data = trainData, kernel = "linear")
    svm_pred <- predict(svm_model, testData)
    svm_pred <- factor(svm_pred, levels = target_levels)
    svm_metrics <- evaluate_model(svm_pred, testData$official_video)
    metrics_list[["SVM"]] <- svm_metrics
    
    # 5. K-Means Clustering
    set.seed(123)
    kmeans_data <- scale(df[, -10])  # Scale the data (exclude the target variable)
    kmeans_model <- kmeans(kmeans_data, centers = 2, nstart = 25)
    cluster_labels <- ifelse(kmeans_model$cluster == 1, "True", "False")
    cluster_labels <- factor(cluster_labels, levels = target_levels)
    kmeans_metrics <- evaluate_model(cluster_labels, df$official_video)
    metrics_list[["K-Means"]] <- kmeans_metrics
    
    df$Cluster <- factor(kmeans_model$cluster)  # Assign clusters
    
    list(metrics_list = metrics_list, df = df, kmeans_model = kmeans_model, 
         testData = testData, knn_pred = knn_pred, nb_pred = nb_pred, 
         dt_pred = dt_pred, svm_pred = svm_pred, cluster_labels = cluster_labels)
  })
  
  # Function to plot confusion matrix
  plot_confusion_matrix <- function(predicted, actual, model_name) {
    cm <- table(Predicted = predicted, Actual = actual)
    ggplot(data = as.data.frame(cm), aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white", size = 6) +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(
        title = paste("Confusion Matrix -", model_name),
        x = "True Label",
        y = "Predicted Label"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10)
      )
  }
  
  # Render the confusion matrix plot
  output$conf_matrix_plot <- renderPlot({
    metrics_data <- results()
    cm_plots <- list(
      KNN = plot_confusion_matrix(metrics_data$knn_pred, metrics_data$testData$official_video, "KNN"),
      NaiveBayes = plot_confusion_matrix(metrics_data$nb_pred, metrics_data$testData$official_video, "Naive Bayes"),
      DecisionTree = plot_confusion_matrix(metrics_data$dt_pred, metrics_data$testData$official_video, "Decision Tree"),
      SVM = plot_confusion_matrix(metrics_data$svm_pred, metrics_data$testData$official_video, "SVM"),
      KMeans = plot_confusion_matrix(metrics_data$cluster_labels, metrics_data$df$official_video, "K-Means")
    )
    
    grid.arrange(cm_plots$KNN, cm_plots$NaiveBayes, cm_plots$DecisionTree,
                 cm_plots$SVM, cm_plots$KMeans, ncol = 3)
  })
  
  # Render the model metrics table
  output$metrics_table <- renderTable({
    metrics_data <- results()
    metrics_df <- data.frame(
      Model = character(),
      Accuracy = numeric(),
      Error_Rate = numeric(),
      Precision = numeric(),
      Recall = numeric(),
      F1_Score = numeric(),
      stringsAsFactors = FALSE
    )
    
    for (model_name in names(metrics_data$metrics_list)) {
      metrics <- metrics_data$metrics_list[[model_name]]
      new_row <- data.frame(
        Model = model_name,
        Accuracy = metrics["Accuracy"],
        Error_Rate = metrics["Error Rate"],
        Precision = metrics["Precision"],
        Recall = metrics["Recall"],
        F1_Score = metrics["F1 Score"],
        stringsAsFactors = FALSE
      )
      metrics_df <- rbind(metrics_df, new_row)
    }
    metrics_df
  })
  
  # Render K-Means Clustering Visualization plot
  output$kmeans_plot <- renderPlot({
    metrics_data <- results()
    
    ggplot(metrics_data$df, aes(x = Danceability, y = Energy, color = Cluster)) +
      geom_point(size = 2, alpha = 0.8) +
      labs(
        title = "K-Means Clustering Visualization",
        x = "Danceability",
        y = "Energy",
        color = "Cluster"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        axis.title = element_text(size = 12)
      )
  })
  
  # Render the evaluation metrics plot (bar plot) with percentage labels
  output$metrics_plot <- renderPlot({
    metrics_data <- results()
    metrics_df <- data.frame(
      Model = rep(names(metrics_data$metrics_list), each = 5),
      Metric = rep(c("Accuracy", "Error Rate", "Precision", "Recall", "F1 Score"), times = length(metrics_data$metrics_list)),
      Value = unlist(lapply(metrics_data$metrics_list, function(x) x))
    )
    
    # Convert to percentage for displaying
    metrics_df$Percentage <- metrics_df$Value * 100
    
    ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
      geom_bar(stat = "identity", position = "dodge") +
      facet_wrap(~ Metric, scales = "free_y") +
      geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
                position = position_dodge(width = 0.8), vjust = -0.5) +
      labs(
        title = "Model Evaluation Metrics",
        y = "Score",
        x = "Model"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        axis.title = element_text(size = 12)
      )
  })
  
  # Render ROC curve plot for Naive Bayes and Decision Tree
  output$roc_curve_plot <- renderPlot({
    metrics_data <- results()
    testData <- metrics_data$testData
    target_levels <- c("True", "False")
    
    # 1. Naive Bayes ROC Curve
    nb_model <- naive_bayes(official_video ~ ., data = testData)
    nb_prob <- predict(nb_model, testData, type = "prob")
    nb_roc <- roc(testData$official_video, nb_prob[, 2], levels = target_levels)
    
    # 2. Decision Tree ROC Curve
    dt_model <- rpart(official_video ~ ., data = testData, method = "class")
    dt_prob <- predict(dt_model, testData, type = "prob")
    dt_roc <- roc(testData$official_video, dt_prob[, 2], levels = target_levels)
    
    # Plot both ROC curves
    plot(nb_roc, col = "blue", lwd = 2, main = "ROC Curves for Naive Bayes and Decision Tree")
    lines(dt_roc, col = "red", lwd = 2)
    legend("bottomright", legend = c("Naive Bayes", "Decision Tree"), col = c("blue", "red"), lwd = 2)
  })
}

# Run the application
shinyApp(ui = ui, server = server)