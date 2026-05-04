'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'

export function AnatomyCard() {
  const lobeData = [
    { name: 'Frontal', value: 50, fill: '#003366' },
    { name: 'Subcortical', value: 37, fill: '#0066CC' },
    { name: 'Cingulate', value: 6, fill: '#4D94FF' },
    { name: 'Other', value: 7, fill: '#99CCFF' },
  ]

  const topRegions = [
    { region: 'Superior Frontal Gyrus', percentage: '34' },
    { region: 'Middle Frontal Gyrus', percentage: '28' },
    { region: 'Caudate', percentage: '19' },
    { region: 'Anterior Cingulate', percentage: '12' },
    { region: 'Putamen', percentage: '7' },
  ]

  return (
    <Card className="border border-gray-200 bg-white">
      <CardHeader>
        <CardTitle>Anatomical Context</CardTitle>
        <CardDescription>Tumor location based on Harvard-Oxford atlas</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={lobeData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={2} dataKey="value">
              {lobeData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div className="space-y-2">
          <p className="text-sm font-semibold text-gray-700">Top Regions Involved</p>
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2 text-xs font-medium text-gray-600">
              <span>Region</span>
              <span className="text-right">% of tumor</span>
            </div>
            {topRegions.map((item) => (
              <div key={item.region} className="grid grid-cols-2 gap-2 text-sm text-gray-700">
                <span>{item.region}</span>
                <span className="text-right font-mono">{item.percentage}%</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
