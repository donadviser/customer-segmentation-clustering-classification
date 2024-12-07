from fastapi import FastAPI, Request
from typing import Optional
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from marketing import logging
from marketing.pipeline.training_pipeline import TrainPipeline
from marketing.components.model_predictor import PredictionPipeline, MarketingData
from marketing.constants import APP_HOST, APP_PORT

# Create FastAPI app instance
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create Jinja2 templates instance
templates = Jinja2Templates(directory="templates")

# Enable CORS for all origins
origins = ["*"]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class to handle form data
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.i_d: Optional[str] = None
        self.year_birth: Optional[int] = None
        self.education: Optional[str] = None
        self.marital_status: Optional[str] = None
        self.income: Optional[int] = None
        self.kidhome: Optional[int] = None
        self.teenhome: Optional[str] = None
        self.dt_customer: Optional[object] = None
        self.recency: Optional[str] = None
        self.mnt_wines: Optional[str] = None
        self.mnt_fruits: Optional[str] = None
        self.mnt_meat_products: Optional[str] = None
        self.mnt_fish_products: Optional[int] = None
        self.mnt_sweet_products: Optional[int] = None
        self.mnt_gold_prods: Optional[int] = None
        self.num_deals_purchases: Optional[int] = None
        self.num_web_purchases: Optional[int] = None
        self.num_catalog_purchases: Optional[int] = None
        self.num_store_purchases: Optional[int] = None
        self.num_web_visits_month: Optional[float] = None
        self.accepted_cmp1: Optional[float] = None
        self.accepted_cmp2: Optional[float] = None
        self.accepted_cmp3: Optional[float] = None
        self.accepted_cmp4: Optional[str] = None
        self.accepted_cmp5: Optional[str] = None
        self.complain: Optional[str] = None
        self.z_cost_contact: Optional[float] = None
        self.z_revenue: Optional[float] = None
        self.response: Optional[float] = None


    async def get_marketing_data(self):
        """
        Fetch data from a request form and assign values to object attributes
        based on the provided form field mappings.
        """
        form = await self.request.form()
        self.i_d = form.get("i_d")
        self.year_birth = form.get("year_birth")
        self.education = form.get("education")
        self.marital_status = form.get("marital_status")
        self.income = form.get("income")
        self.kidhome = form.get("kidhome")
        self.teenhome = form.get("teenhome")
        self.dt_customer = form.get("dt_customer")
        self.recency = form.get("recency")
        self.mnt_wines = form.get("mnt_wines")
        self.mnt_fruits = form.get("mnt_fruits")
        self.mnt_meat_products = form.get("mnt_meat_products")
        self.mnt_fish_products = form.get("mnt_fish_products")
        self.mnt_sweet_products = form.get("mnt_sweet_products")
        self.mnt_gold_prods = form.get("mnt_gold_prods")
        self.num_deals_purchases = form.get("num_deals_purchases")
        self.num_web_purchases = form.get("num_web_purchases")
        self.num_catalog_purchases = form.get("num_catalog_purchases")
        self.num_store_purchases = form.get("num_store_purchases")
        self.num_web_visits_month = form.get("num_web_visits_month")
        self.accepted_cmp1 = form.get("accepted_cmp1")
        self.accepted_cmp2 = form.get("accepted_cmp2")
        self.accepted_cmp3 = form.get("accepted_cmp3")
        self.accepted_cmp4 = form.get("accepted_cmp4")
        self.accepted_cmp5 = form.get("accepted_cmp5")
        self.complain = form.get("complain")
        self.z_cost_contact = form.get("z_cost_contact")
        self.z_revenue = form.get("z_revenue")
        self.response = form.get("response")


# Route to trigger the training pipeline
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response({"status_code": 200, 'message': "Success"})
    except Exception as e:
        logging.error(f"Error training the pipeline: {str(e)}")
        return Response(f"status_code=500: Error occured {e}")


# Route to render the prediction form
@app.get("/predict")
async def predictGetRouteClient(request: Request):
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": "Rendering"}
        )
    except Exception as e:
        logging.error(f"Error rendering the prediction form: {str(e)}")
        return Response(f"status_code=500: Error occured: {e}")


@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_marketing_data()

        input_data = [
            form.i_d,
            form.year_birth,
            form.education,
            form.marital_status,
            form.income,
            form.kidhome,
            form.teenhome,
            form.dt_customer,
            form.recency,
            form.mnt_wines,
            form.mnt_fruits,
            form.mnt_meat_products,
            form.mnt_fish_products,
            form.mnt_sweet_products,
            form.mnt_gold_prods,
            form.num_deals_purchases,
            form.num_web_purchases,
            form.num_catalog_purchases,
            form.num_store_purchases,
            form.num_web_visits_month,
            form.accepted_cmp1,
            form.accepted_cmp2,
            form.accepted_cmp3,
            form.accepted_cmp4,
            form.accepted_cmp5,
            form.complain,
            form.z_cost_contact,
            form.z_revenue,
            form.response 
        ]

        prediction_pipeline = PredictionPipeline()

        logging.info("To get the input data")
        cost_df = prediction_pipeline.prepare_input_data(input_data=input_data)
        #logging.info(f"Obtained the input data: {cost_df.head()}")
        logging.info(f"cost_df['i_d']: {cost_df['i_d']}")
        logging.info(f"cost_df['education']: {cost_df['education']}")
        logging.info(f"cost_df['dt_customer']: {cost_df['dt_customer']}")
        cost_predictor = CostPredictor()
        cost_value = cost_predictor.predict(X=cost_df)
        cost_value = int(cost_value[0][0])
        predicted_result = cost_value
        logging.info(f"cost_value: {cost_value} | predicted_result: {predicted_result}")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context":  predicted_result}
        )
    except Exception as e:
        logging.error(f"Error predicting the cost: {str(e)}")
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)


