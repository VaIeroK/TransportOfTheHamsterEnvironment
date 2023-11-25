from fastapi import APIRouter
from views.returns import get_data, processed_data_avg

graph_router = APIRouter()


@graph_router.get('/full_data')
async def graph(skip: int = 0, take: int = 50):
    return await get_data(skip, take)


@graph_router.get('/total_count')
async def transport_count():
    data = await get_data(0, 2)
    total_counts = {
        'total_quantity_car': data[0]['total_quantity_car'],
        'total_quantity_van': data[0]['total_quantity_van'],
        'total_quantity_bus': data[0]['total_quantity_bus']
    }
    return total_counts


@graph_router.get('/avg_speed')
async def transport_speed():
    return processed_data_avg
