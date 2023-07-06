import models from '../../../../models.json'
import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json(models)
}
