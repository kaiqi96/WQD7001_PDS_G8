#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(randomForest)

model_rf_full <- readRDS("model_rf_full.rds")


# Define UI
ui <- fluidPage(
  titlePanel("Diabetes Predictor"),
  sidebarLayout(
    sidebarPanel(
      selectInput("gender", "Gender", choices = c("Male" = "Male", "Female" = "Female")),
      selectInput("polyuria", "Polyuria", choices = c("No" = 0, "Yes" = 1)),
      selectInput("polydipsia", "Polydipsia", choices = c("No" = 0, "Yes" = 1)),
      selectInput("sudden_weight_loss", "Sudden Weight Loss", choices = c("No" = 0, "Yes" = 1)),
      selectInput("polyphagia", "Polyphagia", choices = c("No" = 0, "Yes" = 1)),
      selectInput("irritability", "Irritability", choices = c("No" = 0, "Yes" = 1)),
      selectInput("partial_paresis", "Partial Paresis", choices = c("No" = 0, "Yes" = 1)),
      actionButton("predictButton", "Predict")
    ),
    mainPanel(
      textOutput("predictionText")
    )
  )
)



server <- function(input, output, session) {
  # Predict function
  predict_diabetes <- function(data, model_rf_full) {
    
    data_formatted <- data.frame(
      gender = factor(data$gender, levels = c("Male", "Female")),
      polyuria = as.integer(data$polyuria),
      polydipsia = as.integer(data$polydipsia),
      sudden_weight_loss = as.integer(data$sudden_weight_loss),
      polyphagia = as.integer(data$polyphagia),
      irritability = as.integer(data$irritability),
      partial_paresis = as.integer(data$partial_paresis)
    )
    
    predictions <- predict(model_rf_full, data_formatted, type = "response")
    
    return(ifelse(predictions >= 0.7, "You might have diabetes, kindly go for body checkup!", "You might not have diabetes."))
  }
  
  input_data <- reactiveValues()
  
  observeEvent(input$predictButton, {
    input_data$df <- data.frame(
      gender = input$gender,
      polyuria = input$polyuria, 
      polydipsia = input$polydipsia,
      sudden_weight_loss = input$sudden_weight_loss,
      polyphagia = input$polyphagia,
      irritability = input$irritability,
      partial_paresis = input$partial_paresis
    )
    
    output$predictionText <- renderText({
      predict_diabetes(input_data$df, model_rf_full)
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
